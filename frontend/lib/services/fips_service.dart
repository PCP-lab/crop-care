import 'package:latlong2/latlong.dart';
import 'package:crop_care/widgets/disease_map.dart';
import 'package:crop_care/utils/county_index.dart';

class FipsService {
  static CountyIndex? _index;

  /// Initialize with loaded county polygons
  static void init(List<CountyPolygon> counties) {
    _index = CountyIndex(counties);
  }

  /// Whether the FIPS index has been initialized and ready to use
  static bool get isReady => _index != null;

  /// Lookup FIPS + county name for given lat/lon
  static ({String fips, String? name})? fromLatLon(double lat, double lon) {
    final idx = _index;
    if (idx == null) return null; // not initialized
    final county = idx.lookup(LatLng(lat, lon));
    if (county == null) return null;
    return (fips: county.fips, name: county.name);
  }
}