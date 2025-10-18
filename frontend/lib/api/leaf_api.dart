import 'dart:async' show TimeoutException;
import 'dart:convert';
import 'dart:io' show File, HttpException, Platform;
import 'package:flutter/foundation.dart' show kIsWeb, kDebugMode, kProfileMode;
import 'package:http/http.dart' as http;
import 'package:meta/meta.dart';

/// Optional override via: --dart-define=API_BASE_URL=https://your-host:8000
const String _envBaseUrl = String.fromEnvironment('API_BASE_URL', defaultValue: '');

/// Resolve the correct base URL for each runtime.
/// - Web (dev on your Mac): http://localhost:8000
/// - iOS Simulator / macOS app: http://127.0.0.1:8000
/// - Android emulator: http://10.0.2.2:8000
/// - Real devices: set --dart-define=API_BASE_URL or change below to your LAN IP / tunnel.
String get kBaseUrl {
  if (_envBaseUrl.isNotEmpty) return _envBaseUrl;
  if (kIsWeb) return 'http://localhost:8000';
  if (Platform.isIOS || Platform.isMacOS) return 'http://127.0.0.1:8000';
  if (Platform.isAndroid) return 'http://10.0.2.2:8000';
  return 'http://127.0.0.1:8000';
}

@immutable
class TopPrediction {
  final String label;
  final double finalProb; // 0..1
  final int rank;

  const TopPrediction({
    required this.label,
    required this.finalProb,
    required this.rank,
  });

  factory TopPrediction.fromJson(Map<String, dynamic> j) => TopPrediction(
    label: (j['label'] ?? 'Unknown').toString(),
    finalProb: (j['final'] is num) ? (j['final'] as num).toDouble() : 0.0,
    rank: (j['rank'] is num) ? (j['rank'] as num).toInt() : 0,
  );
}

@immutable
class PredictResponse {
  final List<TopPrediction> top;
  final String? overlayBase64;
  final double? leafPct; // 0..100 from backend
  final double? spotPct; // 0..100 from backend

  const PredictResponse({
    required this.top,
    this.overlayBase64,
    this.leafPct,
    this.spotPct,
  });

  /// 0..1 convenience getters
  double? get leafFraction => leafPct == null ? null : (leafPct!.clamp(0, 100) / 100.0);
  double? get spotFraction => spotPct == null ? null : (spotPct!.clamp(0, 100) / 100.0);

  factory PredictResponse.fromJson(Map<String, dynamic> j) => PredictResponse(
    top: ((j['top'] as List?) ?? [])
        .whereType<Map<String, dynamic>>()
        .map(TopPrediction.fromJson)
        .toList(),
    overlayBase64: j['overlay_png_base64'] as String?,
    leafPct: (j['leaf_pct'] is num) ? (j['leaf_pct'] as num).toDouble() : null,
    spotPct: (j['spot_pct'] is num) ? (j['spot_pct'] as num).toDouble() : null,
  );
}

class LeafApi {
  LeafApi._();

  // ───────────────────────── IMAGE/TEXT DIAGNOSIS ─────────────────────────
  /// IMAGE/TEXT diagnosis → POST /predict  (multipart/form-data)
  ///
  /// Provide either [imageFile] (mobile/desktop) **or** [imageBytes]+[filename] (web).
  static Future<PredictResponse> predict({
    File? imageFile,
    List<int>? imageBytes,
    String? filename, // required if using [imageBytes]
    String? text,
    bool includeOverlay = true,
    double? latitude,
    double? longitude,
    DateTime? dateNoticedUtc,
    String? dateNoticedIso, // optional raw ISO if you already have it
    Duration timeout = const Duration(seconds: 60),
    int retries = 0, // simple retry on 5xx/timeouts
    http.Client? client,
  }) async {
    assert(
    imageFile != null || (imageBytes != null && filename != null),
    'Provide either imageFile OR (imageBytes + filename)',
    );

    final uri = Uri.parse('$kBaseUrl/predict');

    // Normalize date to ISO-8601 Z (UTC), field name MUST be 'dateNoticed'
    String? isoDate;
    if (dateNoticedIso != null && dateNoticedIso.isNotEmpty) {
      isoDate = dateNoticedIso;
    } else if (dateNoticedUtc != null) {
      isoDate = dateNoticedUtc.toUtc().toIso8601String().replaceFirst(RegExp(r'\.\d+Z$'), 'Z');
    }

    final req = http.MultipartRequest('POST', uri)
      ..fields['include_overlay'] = includeOverlay ? 'true' : 'false';

    if (text != null && text.trim().isNotEmpty) {
      req.fields['text'] = text.trim();
    }
    if (latitude != null && longitude != null) {
      req.fields['latitude'] = latitude.toString();
      req.fields['longitude'] = longitude.toString();
    }
    if (isoDate != null && isoDate.isNotEmpty) {
      req.fields['dateNoticed'] = isoDate;
    }

    // Attach image
    if (imageFile != null) {
      req.files.add(await http.MultipartFile.fromPath('image', imageFile.path));
    } else {
      req.files.add(http.MultipartFile.fromBytes('image', imageBytes!, filename: filename));
    }

    _log('POST /predict (multipart) fields:\n'
        '${const JsonEncoder.withIndent("  ").convert(req.fields)}\n'
        'file: ${imageFile?.path ?? filename}');

    return _withClient(client, (c) async {
      final streamed = await _withRetry(
            () => c.send(req).timeout(timeout),
        retries: retries,
      );
      final res = await http.Response.fromStream(streamed);
      final body = _decodeJsonBody(res);

      if (res.statusCode != 200) {
        throw HttpException('Predict failed (${res.statusCode}): ${body?['error'] ?? res.body}');
      }
      if (body is! Map<String, dynamic>) {
        throw const HttpException('Predict failed: invalid JSON response');
      }
      return PredictResponse.fromJson(body);
    });
  }

  // ───────────────────────── FORECAST (7/14 SUPPORT) ─────────────────────────
  /// FORECAST-ONLY → POST /predict-forecast (application/x-www-form-urlencoded)
  ///
  /// The backend supports:
  /// - `windows`: CSV of windows to compute (e.g., "7,14")
  /// - `window_days`: which window fills legacy fields (defaults to 14)
  /// - `focus_label`: disease to align management tips to (case-insensitive)
  ///
  /// Response (new servers):
  /// {
  ///   "windows_used":[7,14],
  ///   "selected_window":14,
  ///   "by_window": {
  ///     "7": { "top":[...], "daily":[...], "management":{...}, ... },
  ///     "14": { ... }
  ///   },
  ///   // legacy fields mirror the selected window:
  ///   "top":[...], "daily":[...], "management":{...}, ...
  /// }
  ///
  /// Response (older servers): legacy fields only. This client will still return the raw map.
  static Future<Map<String, dynamic>> predictForecast({
    required double lat,
    required double lon,
    String timezone = 'UTC',
    String? stateCode,                 // e.g. "GA" (optional)
    int forecastDays = 16,             // backend caps ~16
    String? focusLabel,                // detected disease to align tips
    String? windows,                   // e.g. "7,14" (compute both at once)
    int? windowDays,                   // e.g. 14 (which window populates legacy fields)
    Duration timeout = const Duration(seconds: 30),
    int retries = 0,
    http.Client? client,
  }) async {
    final uri = Uri.parse('$kBaseUrl/predict-forecast');

    final body = <String, String>{
      'latitude': lat.toString(),
      'longitude': lon.toString(),
      'timezone': timezone,
      'forecast_days': forecastDays.toString(),
      if (stateCode != null && stateCode.isNotEmpty) 'state_code': stateCode,
      if (focusLabel != null && focusLabel.trim().isNotEmpty) 'focus_label': focusLabel.trim(),
      if (windows != null && windows.trim().isNotEmpty) 'windows': windows.trim(),
      if (windowDays != null) 'window_days': windowDays.toString(),
    };

    _log('POST /predict-forecast body: ${const JsonEncoder.withIndent("  ").convert(body)}');

    return _withClient(client, (c) async {
      final res = await _withRetry(
            () => c
            .post(
          uri,
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'User-Agent': _ua(),
          },
          body: body,
        )
            .timeout(timeout),
        retries: retries,
      );

      final jsonMap = _decodeJsonBody(res);
      if (res.statusCode != 200) {
        throw HttpException('predictForecast failed (${res.statusCode}): ${jsonMap?['error'] ?? res.body}');
      }
      if (jsonMap is! Map<String, dynamic>) {
        throw const HttpException('predictForecast failed: invalid JSON response');
      }
      return jsonMap;
    });
  }

  // ───────────────────────── helpers ─────────────────────────

  static Future<T> _withClient<T>(http.Client? client, Future<T> Function(http.Client) fn) async {
    final c = client ?? http.Client();
    try {
      return await fn(c);
    } finally {
      if (client == null) c.close();
    }
  }

  /// Simple retry for timeouts / 5xx
  static Future<T> _withRetry<T>(Future<T> Function() op, {int retries = 0}) async {
    int attempt = 0;
    while (true) {
      attempt++;
      try {
        return await op();
      } on HttpException catch (e) {
        if (attempt > retries) rethrow;
        // rethrow on client errors (4xx)
        if (e.message.contains('(4')) rethrow;
        await Future.delayed(Duration(milliseconds: 200 * attempt));
      } on TimeoutException {
        if (attempt > retries) rethrow;
        await Future.delayed(Duration(milliseconds: 200 * attempt));
      } on http.ClientException {
        if (attempt > retries) rethrow;
        await Future.delayed(Duration(milliseconds: 200 * attempt));
      }
    }
  }

  static dynamic _decodeJsonBody(http.Response res) {
    try {
      final text = utf8.decode(res.bodyBytes); // always decode as UTF-8
      if (text.isEmpty) return null;
      return jsonDecode(text);
    } catch (_) {
      return null;
    }
  }

  static void _log(String msg) {
    if (kDebugMode || kProfileMode) {
      // ignore: avoid_print
      print(msg); // debugPrint truncates long lines
    }
  }

  static String _ua() => 'CropCare/1.0 (${kIsWeb ? "web" : (Platform.operatingSystem)})';
}