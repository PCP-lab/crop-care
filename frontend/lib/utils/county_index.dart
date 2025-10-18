import 'package:latlong2/latlong.dart';
import 'package:crop_care/widgets/disease_map.dart'; // or wherever CountyPolygon is defined

/// --- Point-in-polygon (ray casting) ---
bool _pointInPolygon(LatLng point, List<LatLng> polygon) {
  final x = point.longitude;
  final y = point.latitude;
  bool inside = false;

  for (int i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    final xi = polygon[i].longitude, yi = polygon[i].latitude;
    final xj = polygon[j].longitude, yj = polygon[j].latitude;

    final intersect = ((yi > y) != (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) == 0 ? 1e-12 : (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

/// Reusable index for county lookup by lat/lon
class CountyIndex {
  final List<CountyPolygon> _counties;

  CountyIndex(this._counties);

  /// Find the county polygon containing this point.
  /// Falls back to nearest centroid if not inside any polygon.
  CountyPolygon? lookup(LatLng p) {
    // 1) Exact containment check
    for (final c in _counties) {
      if (_pointInPolygon(p, c.outer)) return c;
    }

    // 2) Fallback: nearest centroid
    const dist = Distance();
    double best = double.infinity;
    CountyPolygon? bestC;

    for (final c in _counties) {
      final d = dist.as(LengthUnit.Kilometer, p, c.centroid);
      if (d < best) {
        best = d;
        bestC = c;
      }
    }
    return bestC;
  }
}