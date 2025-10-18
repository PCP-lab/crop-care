// lib/screens/home_screen.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import 'package:crop_care/widgets/custom_app_bar.dart';
import 'package:crop_care/screens/diagnosis_screen.dart';
import 'package:crop_care/l10n/app_localizations.dart';
import 'package:crop_care/widgets/disease_map.dart'; // uses new date picker API

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final String _plantName = 'Hydrangea';
  final int _moisture = 68;

  // richer per-county details for bottom sheet
  final Map<String, _CountyDetails> _detailsByFips = {};

  Future<Map<String, int>> _loadCounts(DateTime? from, DateTime? to) async {
    final col = FirebaseFirestore.instance
        .collection('disease_reports')
        .withConverter<Map<String, dynamic>>(
      fromFirestore: (s, _) => s.data() ?? {},
      toFirestore: (m, _) => m,
    );

    Query<Map<String, dynamic>> q = col;
    if (from != null) {
      q = q.where('date', isGreaterThanOrEqualTo: Timestamp.fromDate(from));
    }
    if (to != null) {
      q = q.where('date', isLessThan: Timestamp.fromDate(to)); // exclusive
    }

    final snap = await q.get();

    final counts = <String, int>{};
    final tmp = <String, _CountyDetails>{};

    for (final doc in snap.docs) {
      final d = doc.data();
      final fips = (d['county_fips'] ?? '').toString().padLeft(5, '0');
      if (fips.isEmpty) continue;

      final disease = (d['disease'] ?? 'Unknown').toString();
      final countyName = (d['county_name'] ?? '').toString();

      counts[fips] = (counts[fips] ?? 0) + 1;

      final det = tmp[fips] ?? _CountyDetails(countyName, {});
      det.byDisease[disease] = (det.byDisease[disease] ?? 0) + 1;
      tmp[fips] = det;
    }

    setState(() {
      _detailsByFips
        ..clear()
        ..addAll(tmp);
    });

    return counts;
  }

  void _showCountyBottomSheet(
      BuildContext context,
      String countyName,
      Map<String, int> byDisease, {
        required bool noRecent,
      }) {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (_) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(countyName, style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 6),
              if (noRecent)
                const Text('No diseases registered for this period.')
              else
                ...byDisease.entries.map(
                      (e) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(child: Text(e.key, overflow: TextOverflow.ellipsis)),
                        Text('${e.value}'),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final size = MediaQuery.of(context).size;
    final w = size.width;

    final heroAspect = 16 / 9;
    final heroHeight = math.max(200.0, w / heroAspect);
    final cardHeight = math.max(96.0, w * 0.24);
    final imageWidth = math.max(84.0, w * 0.22);
    final innerHeight = math.max(68.0, cardHeight - 24);
    final textTheme = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;

    // initial range (optional)
    final now = DateTime.now();
    final initialFrom = DateTime(now.year, now.month, now.day).subtract(const Duration(days: 6));
    final initialTo = DateTime(now.year, now.month, now.day).add(const Duration(days: 1));

    return Scaffold(
      appBar: CustomAppBar(title: l10n.homeTitle),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: ConstrainedBox(
            constraints: BoxConstraints(
              minHeight: size.height - kToolbarHeight - MediaQuery.of(context).padding.top,
            ),
            child: IntrinsicHeight(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Text(
                    l10n.homeQuestion,
                    style: textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.2,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),

                  // Hero image
                  ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: SizedBox(
                      width: double.infinity,
                      height: heroHeight,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.asset('assets/home/images/homePage_image.jpg', fit: BoxFit.cover),
                          Container(
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.black.withOpacity(0.20),
                                  Colors.black.withOpacity(0.05),
                                ],
                                begin: Alignment.bottomCenter,
                                end: Alignment.topCenter,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 22),

                  Align(
                    alignment: Alignment.centerLeft,
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4.0),
                      child: Text(
                        l10n.homeInstruction,
                        style: textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                          color: scheme.onSurface.withOpacity(0.85),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),

                  // Plant card
                  _FrostedCard(
                    child: InkWell(
                      borderRadius: BorderRadius.circular(16),
                      onTap: () => Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => const DiagnosisScreen()),
                      ),
                      child: SizedBox(
                        height: cardHeight,
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Row(
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(12),
                                child: SizedBox(
                                  width: imageWidth,
                                  height: innerHeight,
                                  child: FittedBox(
                                    fit: BoxFit.cover,
                                    clipBehavior: Clip.hardEdge,
                                    child: Image.asset('assets/home/plants/hydrangea.jpeg'),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      l10n.plantName(_plantName),
                                      style: textTheme.titleLarge?.copyWith(
                                        fontWeight: FontWeight.w800,
                                        color: scheme.onSurface,
                                      ),
                                    ),
                                    const SizedBox(height: 6),
                                    Row(
                                      children: [
                                        Icon(Icons.water_drop, size: 18, color: scheme.primary),
                                        const SizedBox(width: 6),
                                        Expanded(
                                          child: Text(
                                            l10n.plantFactMoisture(_moisture),
                                            style: textTheme.bodyMedium?.copyWith(
                                              color: scheme.onSurface.withOpacity(0.85),
                                            ),
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 8),
                              Icon(Icons.chevron_right, color: scheme.onSurface.withOpacity(0.7)),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 12),
                  Text(
                    'Tip: better, well-lit leaf photos improve detection.',
                    style: textTheme.bodySmall?.copyWith(
                      color: scheme.onSurface.withOpacity(0.7),
                    ),
                    textAlign: TextAlign.center,
                  ),

                  const SizedBox(height: 16),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Disease trends',
                      style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Select any date or range to view registered diseases on the map.',
                      style: textTheme.bodySmall,
                    ),
                  ),
                  const SizedBox(height: 10),

                  // Map with date picker support
                  _FrostedCard(
                    child: SizedBox(
                      height: 380,
                      width: double.infinity,
                      child: DiseaseMap(
                        loadCounts: _loadCounts,
                        initialCountsByFips: const {},
                        initialFrom: initialFrom,
                        initialTo: initialTo,
                        onTapCounty: (fips, name) {
                          final d = _detailsByFips[fips];
                          final byDis = d?.byDisease ?? const <String, int>{};
                          final empty = byDis.isEmpty;
                          _showCountyBottomSheet(
                            context,
                            d?.countyName ?? name ?? 'This county',
                            byDis,
                            noRecent: empty,
                          );
                        },
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _CountyDetails {
  final String countyName;
  final Map<String, int> byDisease;
  _CountyDetails(this.countyName, this.byDisease);
}

class _FrostedCard extends StatelessWidget {
  final Widget child;
  const _FrostedCard({required this.child});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      decoration: BoxDecoration(
        color: scheme.surface.withOpacity(0.7),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: scheme.outline.withOpacity(0.12)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: ClipRRect(borderRadius: BorderRadius.circular(16), child: child),
    );
  }
}