// lib/utils/save_report.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart' show kDebugMode, debugPrint;

import 'package:crop_care/services/fips_service.dart';

/// Saves a disease report only if the model confidence is high enough
/// and the location can be mapped to a county FIPS.
///
/// Returns true if a document was written, false otherwise.
Future<bool> saveReportIfConfident({
  required double lat,
  required double lon,
  required DateTime dateUtc,
  required String disease,
  required double confidence,

  // Optional metadata (recommended)
  String? stateCode,
  String timezone = 'UTC',
  String app = 'crop_care',
  String source = 'image_top1',
}) async {
  const double kMinConfidence = 0.98;

  // Confidence gate
  if (confidence < kMinConfidence) {
    if (kDebugMode) {
      debugPrint(
        '[saveReport] Skip: confidence ${(confidence * 100).toStringAsFixed(2)}% < ${(kMinConfidence * 100).toStringAsFixed(0)}%',
      );
    }
    return false;
  }

  // FIPS service readiness
  if (!(FipsService.isReady ?? false)) {
    if (kDebugMode) debugPrint('[saveReport] Skip: FIPS service not ready');
    return false;
  }

  // County lookup
  final fipsResult = FipsService.fromLatLon(lat, lon);
  if (fipsResult == null) {
    if (kDebugMode) debugPrint('[saveReport] Skip: FIPS lookup failed for ($lat,$lon)');
    return false;
  }

  // Ensure 5-char FIPS
  final countyFips = fipsResult.fips.toString().padLeft(5, '0');
  final countyName = fipsResult.name ?? '';

  final uid = FirebaseAuth.instance.currentUser?.uid;

  final data = <String, dynamic>{
    'app': app,
    'source': source,

    // Location/time
    'lat': lat,
    'lon': lon,
    'stateCode': stateCode ?? '',
    'timezone': timezone,
    'date': Timestamp.fromDate(dateUtc),
    'createdAt': FieldValue.serverTimestamp(),

    // County keys (required for map)
    'county_fips': countyFips,
    'county_name': countyName,

    // Prediction
    'disease': disease,
    'confidence': confidence, // 0..1
    if (uid != null) 'userId': uid,
  };

  await FirebaseFirestore.instance
      .collection('disease_reports')
      .add(data);

  if (kDebugMode) {
    debugPrint(
      '[saveReport] Wrote disease_reports '
          'FIPS=$countyFips NAME="$countyName" disease="$disease" '
          'conf=${(confidence * 100).toStringAsFixed(1)}% '
          'date=${dateUtc.toIso8601String()}',
    );
  }

  return true;
}