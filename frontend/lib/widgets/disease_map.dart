import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:intl/intl.dart';
import 'package:flutter/foundation.dart' show kDebugMode, debugPrint;
/// ===============================
/// County polygon model + loader
/// ===============================



class CountyPolygon {
  final String fips;
  final String? name;
  final List<LatLng> outer;
  final LatLng centroid;
  CountyPolygon({
    required this.fips,
    required this.outer,
    required this.centroid,
    this.name,
  });
}

Future<List<CountyPolygon>> loadCountiesFromAsset(String assetPath) async {
  final txt = await rootBundle.loadString(assetPath);
  if (kDebugMode) debugPrint('[FIPS] Asset "${assetPath}" length=${txt.length}');

  final json = jsonDecode(txt);

  List features;
  if (json is Map<String, dynamic> && json['features'] is List) {
    features = json['features'] as List;
  } else if (json is List) {
    // Some exports give a bare feature list
    features = json;
  } else {
    if (kDebugMode) debugPrint('[FIPS] Unrecognized JSON root. Type=${json.runtimeType}');
    return const <CountyPolygon>[];
  }

  int total = features.length;
  int noGeom = 0, badGeom = 0, noId = 0, tooSmall = 0, ok = 0;

  LatLng toLL(dynamic c) => LatLng(
    (c[1] as num).toDouble(),
    (c[0] as num).toDouble(),
  );

  LatLng centroid(List<LatLng> pts) {
    double signedArea = 0, cx = 0, cy = 0;
    for (int i = 0; i < pts.length; i++) {
      final j = (i + 1) % pts.length;
      final x0 = pts[i].longitude, y0 = pts[i].latitude;
      final x1 = pts[j].longitude, y1 = pts[j].latitude;
      final a = x0 * y1 - x1 * y0;
      signedArea += a;
      cx += (x0 + x1) * a;
      cy += (y0 + y1) * a;
    }
    if (signedArea.abs() < 1e-12) {
      final avgLat = pts.fold<double>(0, (s, p) => s + p.latitude) / pts.length;
      final avgLon = pts.fold<double>(0, (s, p) => s + p.longitude) / pts.length;
      return LatLng(avgLat, avgLon);
    }
    signedArea *= 0.5;
    cx = cx / (6 * signedArea);
    cy = cy / (6 * signedArea);
    return LatLng(cy, cx);
  }

  String? extractFips(Map<String, dynamic> props) {
    // Try common keys
    final geoId = props['GEOID'] ?? props['geoid'] ?? props['GEOID20'] ?? props['GEOID10'];
    if (geoId != null) return geoId.toString();
    final fips = props['FIPS'] ?? props['fips'];
    if (fips != null) return fips.toString();
    final state = props['STATEFP'] ?? props['STATE'] ?? props['statefp'];
    final county = props['COUNTYFP'] ?? props['COUNTY'] ?? props['countyfp'];
    if (state != null && county != null) {
      return '${state}${county}'.toString().padLeft(5, '0');
    }
    return null;
  }

  String? extractName(Map<String, dynamic> props) {
    return (props['NAME'] ?? props['NAME10'] ?? props['name'] ?? props['NAMELSAD'])?.toString();
  }

  List<CountyPolygon> out = [];

  for (final raw in features) {
    final fMap = (raw is Map) ? raw.cast<String, dynamic>() : <String, dynamic>{};

    final geom = (fMap['geometry'] ?? fMap['geom']) as Map<String, dynamic>?;
    if (geom == null) {
      noGeom++;
      continue;
    }

    final props = (fMap['properties'] ?? fMap['attrs'] ?? <String, dynamic>{})
        .cast<String, dynamic>();
    final fips = extractFips(props);
    final name = extractName(props);

    if (fips == null || fips.isEmpty) {
      noId++;
      continue;
    }

    final type = geom['type'];
    final coords = geom['coordinates'];
    List<LatLng> ring = [];

    try {
      if (type == 'Polygon') {
        final rings = (coords as List).cast<List>();
        if (rings.isNotEmpty) ring = rings.first.map<LatLng>(toLL).toList();
      } else if (type == 'MultiPolygon') {
        final polys = (coords as List).cast<List>();
        if (polys.isNotEmpty) {
          final firstPolyRings = (polys.first as List).cast<List>();
          if (firstPolyRings.isNotEmpty) {
            ring = firstPolyRings.first.map<LatLng>(toLL).toList();
          }
        }
      } else {
        badGeom++;
        continue;
      }
    } catch (_) {
      badGeom++;
      continue;
    }

    if (ring.length < 3) {
      tooSmall++;
      continue;
    }

    out.add(CountyPolygon(
      fips: fips.padLeft(5, '0'),
      name: name,
      outer: ring,
      centroid: centroid(ring),
    ));
    ok++;
  }

  if (kDebugMode) {
    debugPrint('[FIPS] features=$total ok=$ok noGeom=$noGeom badGeom=$badGeom noId=$noId tooSmall=$tooSmall');
  }
  return out;
}

/// ===============================
/// Reusable DiseaseMap widget
/// ===============================

/// ===============================
/// Reusable DiseaseMap widget (LIVE)
/// ===============================



// NOTE: This assumes CountyPolygon + loadCountiesFromAsset already exist above.
// If they are in another file, import them instead.

class DiseaseMap extends StatefulWidget {
  /// The initial counts to show before any user selection (optional).
  final Map<String, int> initialCountsByFips;

  /// Called when the user taps a county (unchanged).
  final void Function(String countyFips, String? countyName)? onTapCounty;

  /// Required: given a [from, to] (inclusive start, exclusive end), return counts by FIPS.
  /// If either is null, treat it as open-ended in your implementation (or constrain as you like).
  final Future<Map<String, int>> Function(DateTime? from, DateTime? to) loadCounts;

  /// Optional: an initial date range to display. If null, the map shows [initialCountsByFips].
  final DateTime? initialFrom;
  final DateTime? initialTo;

  const DiseaseMap({
    super.key,
    required this.loadCounts,
    this.initialCountsByFips = const {},
    this.onTapCounty,
    this.initialFrom,
    this.initialTo,
  });

  @override
  State<DiseaseMap> createState() => _DiseaseMapState();
}

class _DiseaseMapState extends State<DiseaseMap> {
  final MapController _mapController = MapController();
  late Future<List<CountyPolygon>> _countiesF;
  List<CountyPolygon> _counties = [];

  // in-widget counts that update when the user picks dates
  Map<String, int> _countsByFips = {};

  // user-selected range (inclusive start, exclusive end)
  DateTime? _from;
  DateTime? _to;

  bool _loadingCounts = false;
  String? _countError;
  LatLng? _lastTap;

  final _fmt = DateFormat('MMM d, yyyy');

  @override
  void initState() {
    super.initState();
    _countiesF = loadCountiesFromAsset('assets/geo/us_counties_simplified.json');

    // seed state
    _countsByFips = Map.of(widget.initialCountsByFips);
    _from = widget.initialFrom;
    _to = widget.initialTo;

    // if an initial range was provided, fetch using it
    if (_from != null || _to != null) {
      _refreshCounts();
    }
  }

  Future<void> _refreshCounts() async {
    setState(() {
      _loadingCounts = true;
      _countError = null;
    });
    try {
      final data = await widget.loadCounts(_from, _to);
      setState(() => _countsByFips = data);
    } catch (e) {
      setState(() => _countError = 'Failed to load data: $e');
    } finally {
      if (mounted) setState(() => _loadingCounts = false);
    }
  }

  String _rangeLabel() {
    if (_from == null && _to == null) return 'All time';
    if (_from != null && _to != null) {
      final toDisplay = _to!.subtract(const Duration(seconds: 1)); // because we treat _to as exclusive
      return '${_fmt.format(_from!)} → ${_fmt.format(DateTime(toDisplay.year, toDisplay.month, toDisplay.day))}';
    }
    if (_from != null) return 'From ${_fmt.format(_from!)}';
    return 'Until ${_fmt.format(_to!.subtract(const Duration(days: 1)))}';
  }

  Color _fillFor(CountyPolygon c) {
    final n = _countsByFips[c.fips] ?? 0;
    if (n <= 0) return Colors.transparent;
    final steps = n.clamp(1, 10);
    final alpha = (0.10 + 0.05 * steps).clamp(0.10, 0.60);
    return Colors.red.withOpacity(alpha.toDouble());
  }

  void _handleTap(LatLng tap) {
    if (_counties.isEmpty) return;
    CountyPolygon? best;
    double bestKm = double.infinity;
    const dist = Distance();
    for (final c in _counties) {
      final km = dist.as(LengthUnit.Kilometer, tap, c.centroid);
      if (km < bestKm) {
        bestKm = km;
        best = c;
      }
    }
    setState(() => _lastTap = tap);
    if (best != null) {
      if (widget.onTapCounty != null) {
        widget.onTapCounty!(best.fips, best.name);
      } else {
        _showLocalSheet(best.name ?? 'This county', best.fips,
            count: _countsByFips[best.fips] ?? 0);
      }
    }
  }

  // pick a date range
  Future<void> _pickRange() async {
    final now = DateTime.now();
    final firstDate = DateTime(now.year - 5, 1, 1);
    final lastDate = DateTime(now.year + 1, 12, 31);

    final initialDateRange = (_from != null && _to != null)
        ? DateTimeRange(start: _from!, end: _to!.subtract(const Duration(days: 0))) // approx
        : null;

    final picked = await showDateRangePicker(
      context: context,
      firstDate: firstDate,
      lastDate: lastDate,
      initialDateRange: initialDateRange,
      helpText: 'Select date range',
      saveText: 'Apply',
    );

    if (picked == null) return;

    // normalize to [startOfDay, startOfNextDayExclusive]
    final start = DateTime(picked.start.year, picked.start.month, picked.start.day);
    final endExclusive = DateTime(picked.end.year, picked.end.month, picked.end.day)
        .add(const Duration(days: 1));

    setState(() {
      _from = start;
      _to = endExclusive;
    });
    await _refreshCounts();
  }

  // pick a single day
  Future<void> _pickSingleDay() async {
    final now = DateTime.now();
    final firstDate = DateTime(now.year - 5, 1, 1);
    final lastDate = DateTime(now.year + 1, 12, 31);
    final picked = await showDatePicker(
      context: context,
      initialDate: _from ?? now,
      firstDate: firstDate,
      lastDate: lastDate,
      helpText: 'Select a date',
    );
    if (picked == null) return;

    final start = DateTime(picked.year, picked.month, picked.day);
    final endExclusive = start.add(const Duration(days: 1));

    setState(() {
      _from = start;
      _to = endExclusive;
    });
    await _refreshCounts();
  }

  void _clearDates() {
    setState(() {
      _from = null;
      _to = null;
    });
    _refreshCounts();
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<CountyPolygon>>(
      future: _countiesF,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snap.hasError) {
          return Center(child: Text('Failed to load counties: ${snap.error}'));
        }

        _counties = snap.data ?? const [];

        return Column(
          children: [
            // ====== Date controls bar ======
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 6),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      _rangeLabel(),
                      style: Theme.of(context).textTheme.bodyMedium,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _pickRange,
                    icon: const Icon(Icons.date_range),
                    label: const Text('Range'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _pickSingleDay,
                    icon: const Icon(Icons.event),
                    label: const Text('Day'),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    tooltip: 'Clear dates',
                    onPressed: _clearDates,
                    icon: const Icon(Icons.clear),
                  ),
                ],
              ),
            ),

            if (_loadingCounts)
              const LinearProgressIndicator(minHeight: 2),

            if (_countError != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: Text(_countError!, style: TextStyle(color: Colors.red.shade700)),
              ),

            // ====== Map ======
            Expanded(
              child: Stack(
                children: [
                  FlutterMap(
                    mapController: _mapController,
                    options: MapOptions(
                      initialCenter: const LatLng(39.5, -98.35),
                      initialZoom: 4,
                      minZoom: 3.2,
                      maxZoom: 14,
                      interactionOptions: const InteractionOptions(flags: InteractiveFlag.all),
                      onTap: (tapPos, latLng) => _handleTap(latLng),
                    ),
                    children: [
                      TileLayer(
                        urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        userAgentPackageName: 'com.example.crop_care',
                      ),
                      PolygonLayer(
                        polygons: _counties
                            .map((c) => Polygon(
                          points: c.outer,
                          color: _fillFor(c),
                          borderColor: Colors.black.withOpacity(0.10),
                          borderStrokeWidth: 0.5,
                        ))
                            .toList(),
                        polygonCulling: true,
                      ),
                      if (_lastTap != null)
                        MarkerLayer(
                          markers: [
                            Marker(
                              point: _lastTap!,
                              width: 24,
                              height: 24,
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Theme.of(context)
                                      .colorScheme
                                      .primary
                                      .withOpacity(0.9),
                                  shape: BoxShape.circle,
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.25),
                                      blurRadius: 6,
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                    ],
                  ),

                  // Zoom controls
                  Positioned(
                    right: 12,
                    bottom: 12,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _ZoomBtn(
                          icon: Icons.add,
                          tooltip: 'Zoom in',
                          onPressed: () {
                            final cam = _mapController.camera;
                            final newZ = (cam.zoom + 1)
                                .clamp(cam.minZoom ?? 1.0, cam.maxZoom ?? 20.0)
                                .toDouble();
                            _mapController.move(cam.center, newZ);
                          },
                        ),
                        const SizedBox(height: 8),
                        _ZoomBtn(
                          icon: Icons.remove,
                          tooltip: 'Zoom out',
                          onPressed: () {
                            final cam = _mapController.camera;
                            final newZ = (cam.zoom - 1)
                                .clamp(cam.minZoom ?? 1.0, cam.maxZoom ?? 20.0)
                                .toDouble();
                            _mapController.move(cam.center, newZ);
                          },
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }

  void _showLocalSheet(String countyName, String countyFips, {required int count}) {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('$countyName • FIPS $countyFips',
                  style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 6),
              Text(_rangeLabel(), style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 12),
              Text('Total reports: $count'),
            ],
          ),
        );
      },
    );
  }
}

class _ZoomBtn extends StatelessWidget {
  final IconData icon;
  final String? tooltip;
  final VoidCallback onPressed;
  const _ZoomBtn({required this.icon, this.tooltip, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip ?? '',
      child: Material(
        elevation: 2,
        shape: const CircleBorder(),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onPressed,
          child: SizedBox(
            width: 44,
            height: 44,
            child: Icon(icon, size: 22), // ✅ use the passed-in icon
          ),
        ),
      ),
    );
  }
}
