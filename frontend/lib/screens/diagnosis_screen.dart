import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:crop_care/widgets/custom_app_bar.dart';
import 'package:crop_care/screens/results_screen.dart';
import 'package:crop_care/l10n/app_localizations.dart';
import 'package:crop_care/utils/save_report.dart';
// Location + geocoding + HTTP for suggestions
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';
import 'package:http/http.dart' as http;
// Your API
import 'package:crop_care/api/leaf_api.dart';
class LatLon {
  final double lat;
  final double lon;
  const LatLon(this.lat, this.lon);
}

class DiagnosisScreen extends StatefulWidget {
  const DiagnosisScreen({super.key});

  @override
  State<DiagnosisScreen> createState() => _DiagnosisScreenState();
}

class _DiagnosisScreenState extends State<DiagnosisScreen> {
  final _formKey = GlobalKey<FormState>();
  final _descCtrl = TextEditingController();
  final _dateCtrl = TextEditingController();
  final _locCtrl  = TextEditingController();
  final _picker = ImagePicker();
  final FocusNode _locFocus = FocusNode();

  XFile? _picked;
  DateTime? _selectedDate;
  bool _submitting = false;

  // Location state
  bool _locLoading = false;
  double? _lat;
  double? _lon;
  String? _stateCode; // e.g. "GA" for US

  // Typeahead suggestions + cache (label -> coords)
  final _suggestions = <String>[];
  final Map<String, LatLon> _locCache = {};

  // Debounce
  Timer? _debounce;

  @override
  void initState() {
    super.initState();
    _locCtrl.addListener(_onLocationChanged);
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _locCtrl.removeListener(_onLocationChanged);
    _locCtrl.dispose();
    _locFocus.dispose();
    _descCtrl.dispose();
    _dateCtrl.dispose();
    super.dispose();
  }

  void _safeSetState(VoidCallback fn) {
    if (!mounted) return;
    setState(fn);
  }

  // ---------- US state parsing ----------
  String? _usStateFromPlacemark(Placemark pm) {
    final cc = (pm.isoCountryCode ?? '').toUpperCase();
    final admin = (pm.administrativeArea ?? '').trim();
    if (cc == 'US' && admin.length == 2) return admin.toUpperCase();
    return null;
  }

  // ------------ image pickers ------------
  Future<void> _pickFrom(ImageSource source) async {
    final l10n = AppLocalizations.of(context)!;
    try {
      final file = await _picker.pickImage(
        source: source,
        maxWidth: 2048,
        maxHeight: 2048,
        imageQuality: 85,
      );
      if (file != null) {
        _safeSetState(() => _picked = file);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('${l10n.uploadPhoto}: $e')),
      );
    }
  }

  Future<void> _showPickSheet() async {
    final l10n = AppLocalizations.of(context)!;
    final base = Theme.of(context);
    final isDark = base.brightness == Brightness.dark;
    final sheetBg = isDark ? const Color(0xFF0D0D0D) : Colors.white;

    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      backgroundColor: sheetBg,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) {
        final theme = Theme.of(ctx);
        final dark = theme.brightness == Brightness.dark;
        final surface = dark ? const Color(0xFF1A1A1A) : Colors.white;
        final on = dark ? Colors.white : Colors.black87;

        final themed = theme.copyWith(
          listTileTheme: theme.listTileTheme.copyWith(
            iconColor: on,
            textColor: on,
            titleTextStyle: theme.textTheme.bodyLarge?.copyWith(color: on),
          ),
          iconTheme: theme.iconTheme.copyWith(color: on),
          colorScheme: theme.colorScheme.copyWith(
            surface: surface,
            background: sheetBg,
            onSurface: on,
          ),
          splashColor: on.withOpacity(0.08),
          highlightColor: on.withOpacity(0.06),
        );

        return Theme(
          data: themed,
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Wrap(
                runSpacing: 8,
                children: [
                  ListTile(
                    leading: const Icon(Icons.photo_camera_outlined),
                    title: Text(l10n.camera),
                    onTap: () {
                      Navigator.pop(ctx);
                      _pickFrom(ImageSource.camera);
                    },
                  ),
                  ListTile(
                    leading: const Icon(Icons.photo_library_outlined),
                    title: Text(l10n.gallery),
                    onTap: () {
                      Navigator.pop(ctx);
                      _pickFrom(ImageSource.gallery);
                    },
                  ),
                  if (_picked != null)
                    ListTile(
                      leading: const Icon(Icons.delete_outline),
                      title: Text(MaterialLocalizations.of(context).deleteButtonTooltip),
                      onTap: () {
                        _safeSetState(() => _picked = null);
                        Navigator.pop(ctx);
                      },
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  // ------------ date picker ------------
  Future<void> _pickDate() async {
    final now = DateTime.now();
    final oneYearAgo = DateTime(now.year, now.month, now.day)
        .subtract(const Duration(days: 365));

    final picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate ?? now,
      firstDate: oneYearAgo,
      lastDate: now,
      builder: (context, child) {
        final base = Theme.of(context);
        final isDark = base.brightness == Brightness.dark;
        if (!isDark) return child!;

        const darkSurface = Color(0xFF1E1E1E);
        const darkBg = Color(0xFF121212);
        final primary = base.colorScheme.primary;

        return Theme(
          data: ThemeData(
            useMaterial3: true,
            brightness: Brightness.dark,
            colorScheme: base.colorScheme.copyWith(
              brightness: Brightness.dark,
              surface: darkSurface,
              onSurface: Colors.white,
              primary: primary,
              onPrimary: Colors.white,
            ),
            dialogTheme: const DialogThemeData(
              backgroundColor: darkBg,
              surfaceTintColor: Colors.transparent,
            ),
            datePickerTheme: DatePickerThemeData(
              backgroundColor: darkBg,
              surfaceTintColor: Colors.transparent,
              headerBackgroundColor: darkSurface,
              headerForegroundColor: Colors.white,
              dayForegroundColor: MaterialStateProperty.resolveWith((states) {
                if (states.contains(MaterialState.disabled)) {
                  return Colors.white.withOpacity(0.38);
                }
                if (states.contains(MaterialState.selected)) {
                  return Colors.black;
                }
                return Colors.white;
              }),
              dayBackgroundColor: MaterialStateProperty.resolveWith((states) {
                if (states.contains(MaterialState.selected)) return primary;
                return Colors.transparent;
              }),
              todayForegroundColor: const MaterialStatePropertyAll(Colors.white),
              todayBackgroundColor: const MaterialStatePropertyAll(Colors.transparent),
              todayBorder: const BorderSide(color: Colors.white54),
              yearForegroundColor: MaterialStateProperty.resolveWith((states) {
                if (states.contains(MaterialState.selected)) return Colors.black;
                return Colors.white;
              }),
              yearBackgroundColor: MaterialStateProperty.resolveWith((states) {
                if (states.contains(MaterialState.selected)) return primary;
                return Colors.transparent;
              }),
            ),
          ),
          child: child!,
        );
      },
    );

    if (picked != null) {
      final l10n = AppLocalizations.of(context)!;
      final formatted = DateFormat.yMMMd(l10n.localeName).format(picked);
      _safeSetState(() {
        _selectedDate = picked;
        _dateCtrl.text = formatted;
      });
    }
  }

  // ------------ location helpers ------------
  void _onLocationChanged() {
    _lat = null; _lon = null; _stateCode = null;
    final q = _locCtrl.text.trim();
    _debounce?.cancel();
    if (q.length < 3) {
      _safeSetState(() => _suggestions.clear());
      return;
    }
    _debounce = Timer(const Duration(milliseconds: 350), () {
      _searchPlaces(q);
    });
  }

  Future<void> _useCurrentLocation() async {
    _safeSetState(() => _locLoading = true);
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        await Geolocator.openLocationSettings();
        throw Exception('Location services are disabled.');
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied) {
        throw Exception('Location permission denied.');
      }
      if (permission == LocationPermission.deniedForever) {
        await Geolocator.openAppSettings();
        throw Exception('Location permission permanently denied.');
      }

      final pos = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.best,
      );

      _lat = pos.latitude;
      _lon = pos.longitude;

      final label = await _reverseGeocode(_lat!, _lon!); // fills _stateCode
      _safeSetState(() {
        _locCtrl.text = label ?? '${_lat!.toStringAsFixed(5)}, ${_lon!.toStringAsFixed(5)}';
        _suggestions.clear();
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Couldn’t get location: $e')),
      );
    } finally {
      _safeSetState(() => _locLoading = false);
    }
  }

  Future<String?> _reverseGeocode(double lat, double lon) async {
    try {
      final marks = await placemarkFromCoordinates(lat, lon);
      if (marks.isEmpty) return null;
      final pm = marks.first;
      _stateCode = _usStateFromPlacemark(pm);
      final city = pm.locality?.isNotEmpty == true ? pm.locality : pm.subAdministrativeArea;
      final admin = pm.administrativeArea;
      final country = pm.country;
      return [
        if (city != null && city.isNotEmpty) city,
        if (admin != null && admin.isNotEmpty) admin,
        if (country != null && country.isNotEmpty) country,
      ].join(', ');
    } catch (_) {
      return null;
    }
  }

  Future<void> _searchPlaces(String query) async {
    try {
      _safeSetState(() => _locLoading = true);
      final url = Uri.parse(
        'https://nominatim.openstreetmap.org/search'
            '?format=json&addressdetails=1&limit=6&q=${Uri.encodeQueryComponent(query)}',
      );
      final res = await http.get(
        url,
        headers: {
          'User-Agent': 'CropCare/1.0 (contact: youremail@example.com)',
        },
      );
      if (res.statusCode != 200) return;

      final List data = jsonDecode(res.body) as List;
      _suggestions.clear();
      _locCache.clear();

      for (final item in data) {
        final display = item['display_name'] as String? ?? '';
        final lat = double.tryParse(item['lat']?.toString() ?? '');
        final lon = double.tryParse(item['lon']?.toString() ?? '');
        if (display.isNotEmpty && lat != null && lon != null) {
          String label = display;
          final addr = item['address'] as Map<String, dynamic>?;
          if (addr != null) {
            final city = addr['city'] ?? addr['town'] ?? addr['village'] ?? addr['hamlet'];
            final state = addr['state'];
            final country = addr['country'];
            final parts = <String>[];
            if (city != null && (city as String).isNotEmpty) parts.add(city);
            if (state != null && (state as String).isNotEmpty) parts.add(state);
            if (country != null && (country as String).isNotEmpty) parts.add(country);
            if (parts.isNotEmpty) label = parts.join(', ');
          }
          _suggestions.add(label);
          _locCache[label] = LatLon(lat, lon);
        }
      }
      _safeSetState(() {});
    } catch (_) {
      // suggestions are optional
    } finally {
      _safeSetState(() => _locLoading = false);
    }
  }

  Future<void> _geocodeIfNeeded() async {
    final t = _locCtrl.text.trim();
    if (t.isEmpty) {
      _lat = null; _lon = null; _stateCode = null;
      return;
    }
    final cached = _locCache[t];
    if (cached != null) {
      _lat = cached.lat;
      _lon = cached.lon;
      await _reverseGeocode(_lat!, _lon!); // fill _stateCode
      return;
    }
    try {
      _safeSetState(() => _locLoading = true);
      final results = await locationFromAddress(t);
      if (results.isNotEmpty) {
        _lat = results.first.latitude;
        _lon = results.first.longitude;
        await _reverseGeocode(_lat!, _lon!); // fill _stateCode
      } else {
        throw Exception('No matches for “$t”.');
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Couldn’t find that place: $e')),
      );
    } finally {
      _safeSetState(() => _locLoading = false);
    }
  }

  // ------------ submit ------------
  Future<void> _submit() async {
    final l10n = AppLocalizations.of(context)!;
    FocusScope.of(context).unfocus();

    if (_picked == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(l10n.uploadPhoto)));
      return;
    }
    if (!_formKey.currentState!.validate()) return;

    if (_selectedDate == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(l10n.dateNoticed)));
      return;
    }

    await _geocodeIfNeeded();
    if (_lat == null || _lon == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a valid location or use current location')),
      );
      return;
    }

    final imgFile = File(_picked!.path);

    // Pin to midnight UTC (weather pipeline expects date-only)
    final selected = _selectedDate!;
    final dateUtc = DateTime.utc(selected.year, selected.month, selected.day);

    if (kDebugMode) {
      final payloadPreview = <String, dynamic>{
        'text': _descCtrl.text.trim(),
        'latitude': _lat,
        'longitude': _lon,
        'stateCode': _stateCode,
        'dateNoticedUtc': dateUtc.toIso8601String(),
        'includeOverlay': true,
        'image': {
          'path': _picked!.path,
          'bytes': await imgFile.length(),
        },
      };
      final pretty = const JsonEncoder.withIndent('  ').convert(payloadPreview);
      debugPrint('---- Diagnose payload ----\n$pretty');
    }

    _safeSetState(() => _submitting = true);

    try {
      // 1) Run model prediction
      final resp = await LeafApi.predict(
        imageFile: imgFile,
        text: _descCtrl.text.trim(),
        includeOverlay: true,
        latitude: _lat,
        longitude: _lon,
        dateNoticedUtc: dateUtc,
      );

      // 2) Extract top prediction + confidence (from TopPrediction.finalProb 0..1)
      // 2) Extract top prediction + confidence (from TopPrediction.finalProb 0..1)
      final TopPrediction? top = resp.top.isNotEmpty ? resp.top.first : null;
      final String diseaseName = top?.label ?? 'Unknown';
      final double conf = top?.finalProb ?? 0.0;

// 3) Save minimal report only if confidence >= 0.98
      bool saved = false;
      try {
        saved = await saveReportIfConfident(
          lat: _lat!,
          lon: _lon!,
          dateUtc: dateUtc,
          disease: diseaseName,
          confidence: conf,        // already 0..1 per your API model
          stateCode: _stateCode,   // ✅ add this
          timezone: 'UTC',         // ✅ add this (or a local tz)
        );
      } catch (e) {
        if (kDebugMode) debugPrint('saveReportIfConfident failed: $e');
      }

      if (!mounted) return;
      _safeSetState(() => _submitting = false);

// 4) Optional UX feedback (clarify why not saved)
      if (saved) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved: $diseaseName (${(conf * 100).toStringAsFixed(1)}%)')),
        );
      } else {
        final reason = (conf < 0.98)
            ? 'confidence ${(conf * 100).toStringAsFixed(1)}% < 98%'
            : 'location-to-county lookup not ready';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Not saved ($reason).')),
        );
      }

// 5) Navigate to results (unchanged)
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultsScreen(
            predictions: resp.top,
            overlayBase64: resp.overlayBase64,
            spotCoverage: resp.spotFraction,
            lat: _lat,
            lon: _lon,
            stateCode: _stateCode,
            timezone: 'UTC',
            takenAtUtc: dateUtc,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      _safeSetState(() => _submitting = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Diagnosis failed: $e')),
      );
    }
  }

  // ------------ UI ------------
  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final w = MediaQuery.of(context).size.width;
    final textTheme = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: CustomAppBar(title: l10n.diagnose),
      backgroundColor: Theme.of(context).brightness == Brightness.dark ? Colors.black : Colors.white,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(l10n.uploadPhoto, style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                _buildUploadArea(),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => _pickFrom(ImageSource.camera),
                        icon: const Icon(Icons.photo_camera_outlined),
                        label: Text(l10n.camera, style: textTheme.labelLarge),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => _pickFrom(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library_outlined),
                        label: Text(l10n.gallery, style: textTheme.labelLarge),
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 24),

                Text(l10n.description, style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                ConstrainedBox(
                  constraints: BoxConstraints(minHeight: 80, maxHeight: 160, maxWidth: w),
                  child: Scrollbar(
                    child: TextFormField(
                      controller: _descCtrl,
                      maxLines: null,
                      style: textTheme.bodyLarge,
                      decoration: InputDecoration(
                        hintText: l10n.descriptionHint,
                        hintStyle: textTheme.bodyMedium?.copyWith(color: scheme.onSurface.withOpacity(0.6)),
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                        contentPadding: const EdgeInsets.all(12),
                      ),
                      validator: (v) {
                        final t = v?.trim() ?? '';
                        if (t.isEmpty) return l10n.description;
                        if (t.length < 10) return '≥ 10';
                        return null;
                      },
                    ),
                  ),
                ),

                const SizedBox(height: 24),
                Text('Location', style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),

                RawAutocomplete<String>(
                  focusNode: _locFocus,
                  textEditingController: _locCtrl,
                  optionsBuilder: (TextEditingValue te) {
                    const useCurrentToken = '__USE_CURRENT__';
                    final q = te.text.trim().toLowerCase();
                    final filtered = (q.isEmpty)
                        ? _suggestions
                        : _suggestions.where((s) => s.toLowerCase().contains(q)).toList();
                    return <String>[useCurrentToken, ...filtered];
                  },
                  fieldViewBuilder: (ctx, controller, focusNode, onFieldSubmitted) {
                    controller.addListener(() {
                      if (controller.text != _locCtrl.text) {
                        _locCtrl.text = controller.text;
                        _locCtrl.selection = controller.selection;
                      }
                    });

                    return TextField(
                      controller: controller,
                      focusNode: focusNode,
                      style: textTheme.bodyLarge,
                      decoration: InputDecoration(
                        hintText: 'City, State or address',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
                        suffixIcon: _locLoading
                            ? const Padding(
                          padding: EdgeInsets.all(12.0),
                          child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                        )
                            : IconButton(
                          tooltip: 'Clear',
                          icon: const Icon(Icons.clear),
                          onPressed: () {
                            controller.clear();
                            _lat = null; _lon = null; _stateCode = null;
                            _safeSetState(() {});
                          },
                        ),
                      ),
                      onTap: () => _safeSetState(() {}),
                      onSubmitted: (_) async {
                        await _geocodeIfNeeded();
                        _safeSetState(() {});
                      },
                    );
                  },
                  optionsViewBuilder: (context, onSelected, options) {
                    final list = options.toList();
                    if (list.isEmpty) return const SizedBox.shrink();

                    return Align(
                      alignment: Alignment.topLeft,
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 640, maxHeight: 320),
                        child: Material(
                          elevation: 6,
                          borderRadius: BorderRadius.circular(12),
                          color: Theme.of(context).cardColor,
                          child: ListView.separated(
                            padding: EdgeInsets.zero,
                            itemCount: list.length,
                            separatorBuilder: (_, __) => const Divider(height: 1, thickness: 0.6),
                            itemBuilder: (ctx, i) {
                              final item = list[i];
                              if (i == 0 && item == '__USE_CURRENT__') {
                                return ListTile(
                                  leading: const Icon(Icons.my_location_outlined),
                                  title: const Text('Use current location', style: TextStyle(fontWeight: FontWeight.w600)),
                                  subtitle: const Text('Get weather for your current spot'),
                                  onTap: () async {
                                    onSelected(item);
                                    FocusScope.of(context).unfocus();
                                    await _useCurrentLocation();
                                    if (!mounted) return;
                                    _safeSetState(() {});
                                  },
                                );
                              }
                              return ListTile(
                                leading: const Icon(Icons.place_outlined),
                                title: Text(item, maxLines: 1, overflow: TextOverflow.ellipsis),
                                onTap: () async {
                                  onSelected(item);
                                  _locCtrl.text = item;
                                  final coords = _locCache[item];
                                  if (coords != null) {
                                    _lat = coords.lat;
                                    _lon = coords.lon;
                                    await _reverseGeocode(_lat!, _lon!); // set _stateCode
                                  } else {
                                    await _geocodeIfNeeded();
                                  }
                                  if (!mounted) return;
                                  _safeSetState(() {});
                                },
                              );
                            },
                          ),
                        ),
                      ),
                    );
                  },
                  onSelected: (_) {},
                ),

                const SizedBox(height: 24),

                Text(l10n.dateNoticed, style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                TextField(
                  controller: _dateCtrl,
                  readOnly: true,
                  style: textTheme.bodyLarge,
                  decoration: InputDecoration(
                    hintText: l10n.dateNoticed,
                    suffixIcon: IconButton(
                      onPressed: _pickDate,
                      icon: const Icon(Icons.calendar_today_outlined),
                      tooltip: l10n.dateNoticed,
                    ),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
                  ),
                  onTap: _pickDate,
                ),

                const SizedBox(height: 28),

                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _submitting ? null : _submit,
                    icon: _submitting
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.biotech_outlined),
                    label: Text(
                      _submitting ? '${l10n.diagnose}...' : l10n.diagnose,
                      style: textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w600),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildUploadArea() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final borderColor = isDark ? Colors.white24 : Colors.grey;

    final placeholder = Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.cloud_upload_outlined, size: 32),
          const SizedBox(height: 8),
          Text(
            'Upload your photo here',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: isDark ? Colors.white60 : Colors.grey,
            ),
          ),
        ],
      ),
    );

    final content = _picked == null ? placeholder : _buildImagePreview();

    return GestureDetector(
      onTap: _showPickSheet,
      child: DottedBorder(
        color: borderColor,
        strokeWidth: 1,
        dashPattern: const [6, 4],
        borderType: BorderType.RRect,
        radius: const Radius.circular(12),
        child: Container(
          width: double.infinity,
          height: 200,
          padding: const EdgeInsets.all(2),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: content,
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_picked == null) return const SizedBox.shrink();
    if (kIsWeb) {
      return Image.network(
        _picked!.path,
        height: 220,
        width: double.infinity,
        fit: BoxFit.cover,
      );
    } else {
      return Image.file(
        File(_picked!.path),
        height: 220,
        width: double.infinity,
        fit: BoxFit.cover,
      );
    }
  }
}