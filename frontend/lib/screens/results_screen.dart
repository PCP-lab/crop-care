import 'dart:convert';
import 'package:crop_care/screens/weather_prediction_screen.dart';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:crop_care/api/leaf_api.dart';
import 'dart:ui' show FontFeature;

class ResultsScreen extends StatefulWidget {
    final List<TopPrediction> predictions;
    final String? overlayBase64;
    final double? spotCoverage; // 0..1 fraction

    // values carried from Diagnose screen (optional but recommended)
    final double? lat;
    final double? lon;
    final String? stateCode; // e.g. "GA"
    final String timezone; // defaults to 'UTC' below if not passed
    final DateTime? takenAtUtc; // photo date

    const ResultsScreen({
        super.key,
        required this.predictions,
        this.overlayBase64,
        this.spotCoverage,
        this.lat,
        this.lon,
        this.stateCode,
        this.timezone = 'UTC',
        this.takenAtUtc,
    });

    /// Convenience: build from raw API response (image prediction only).
    factory ResultsScreen.fromApiResponse(
        Map<String, dynamic> data, {
            double? lat,
            double? lon,
            String? stateCode,
            String timezone = 'UTC',
            DateTime? takenAtUtc,
        }) {
        final top = (data['top'] as List? ?? [])
            .map((e) => TopPrediction(
            label: e['label'] as String? ?? 'Unknown',
            finalProb: (e['final'] as num?)?.toDouble() ?? 0.0,
            rank: (e['rank'] as num?)?.toInt() ?? 0,
        ))
            .toList();

        final overlay = data['overlay_png_base64'] as String?;

        double? cov;
        final covRaw = data['spot_pct'] ??
            data['spot_coverage'] ??
            data['spot_percent'] ??
            data['coverage'];
        if (covRaw is num) {
            cov = covRaw <= 1.001 ? covRaw.toDouble() : (covRaw.toDouble() / 100.0);
            cov = cov.clamp(0.0, 1.0);
        }

        return ResultsScreen(
            predictions: top,
            overlayBase64: overlay,
            spotCoverage: cov,
            lat: lat,
            lon: lon,
            stateCode: stateCode,
            timezone: timezone,
            takenAtUtc: takenAtUtc,
        );
    }

    @override
    State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
    bool _loadingForecast = false;
    Map<String, dynamic>? _forecastJson; // full response (debug/telemetry)
    bool _loggedThisResult = false;

    // Risk for the detected-top disease over the next week
    List<_DayRisk>? _nextWeekRiskForDetectedTop;

    // Management display model
    _MgmtView? _mgmtView; // built from backend management + peak_risk_for_top

    @override
    void initState() {
        super.initState();
        WidgetsBinding.instance.addPostFrameCallback((_) => _maybeLogTopDetection());
    }

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);

        // Sort explicitly by rank so #1 is always first visually
        final sortedPredictions = [...widget.predictions]..sort((a, b) => a.rank.compareTo(b.rank));
        final top3 = sortedPredictions.take(3).toList();
        final top1 = top3.isNotEmpty ? top3.first : null;

        final overlayWidget =
        (widget.overlayBase64 != null && widget.overlayBase64!.isNotEmpty)
            ? ClipRRect(
            borderRadius: BorderRadius.circular(14),
            child: InteractiveViewer(
                minScale: 0.8,
                maxScale: 4.0,
                child: Image.memory(
                    base64Decode(widget.overlayBase64!),
                    fit: BoxFit.contain,
                ),
            ),
        )
            : Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
                'No overlay returned',
                style: theme.textTheme.bodyMedium,
                textAlign: TextAlign.center,
            ),
        );

        final canForecast = widget.lat != null && widget.lon != null;

        return Scaffold(
            appBar: AppBar(
                title: const Text('Diagnosis Results'),
                elevation: 0,
            ),
            body: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                    // Overlay
                    _SectionHeader('Segmented overlay'),
                    AspectRatio(
                        aspectRatio: 4 / 3,
                        child: Container(
                            decoration: BoxDecoration(
                                border: Border.all(color: theme.colorScheme.outlineVariant),
                                borderRadius: BorderRadius.circular(14),
                            ),
                            child: Center(child: overlayWidget),
                        ),
                    ),

                    // Coverage
                    if (widget.spotCoverage != null) ...[
                        const SizedBox(height: 16),
                        _CoverageStatus(coverage: widget.spotCoverage!),
                    ],

                    // Top-3 predictions
                    const SizedBox(height: 20),
                    _SectionHeader('Top diseases'),
                    const SizedBox(height: 8),
                    if (top3.isEmpty)
                        Text('No predictions', style: theme.textTheme.bodyMedium)
                    else ...[
                        if (top1 != null) _TopPredictionCard(p: top1),
                        ...top3.skip(1).map((p) => _PredictionRow(p: p)),
                    ],

                    // Forecast CTA
                    const SizedBox(height: 24),
                    _PrimaryButton(
                        icon: _loadingForecast
                            ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.storm_outlined),
                        label: _loadingForecast ? 'Loading forecast...' : 'Predict next-week risk + tips',
                        onPressed: (_loadingForecast || widget.lat == null || widget.lon == null)
                            ? null
                            : () async {
                            setState(() => _loadingForecast = true);
                            try {
                                final detected = _detectedTopDiseaseLabel();

                                // 🔥 Fetch both 7 and 14 days in one go
                                final forecastData = await LeafApi.predictForecast(
                                    lat: widget.lat!,
                                    lon: widget.lon!,
                                    timezone: widget.timezone,
                                    stateCode: widget.stateCode,
                                    forecastDays: 16,
                                    windows: '7,14',
                                    windowDays: 7,
                                    focusLabel: detected,
                                );

                                if (!mounted) return;
                                setState(() => _loadingForecast = false);

                                // ✅ Navigate & pass the pre-fetched forecast data
                                Navigator.of(context).push(
                                    MaterialPageRoute(
                                        builder: (_) => WeatherPredictionScreen(
                                            lat: widget.lat!,
                                            lon: widget.lon!,
                                            timezone: widget.timezone,
                                            stateCode: widget.stateCode,
                                            focusLabel: detected,
                                            forecastData: forecastData, // <— pass data here
                                        ),
                                    ),
                                );
                            } catch (e) {
                                if (!mounted) return;
                                setState(() => _loadingForecast = false);
                                ScaffoldMessenger.of(context)
                                    .showSnackBar(SnackBar(content: Text('Forecast failed: $e')));
                            }
                        },
                    ),

                    // Next week risk (detected top disease)
                    if ((_nextWeekRiskForDetectedTop?.isNotEmpty ?? false)) ...[
                        const SizedBox(height: 20),
                        _NextWeekRiskCard(
                            diseaseLabel: _detectedTopDiseaseLabel() ?? 'Detected disease',
                            days: _nextWeekRiskForDetectedTop!,
                        ),
                    ],

                    // Management tips (aligned to peak risk)
                    if (_mgmtView != null) ...[
                        const SizedBox(height: 16),
                        _ManagementCard(view: _mgmtView!),
                    ],
                ],
            ),
        );
    }

    Future<void> _maybeLogTopDetection() async {
        if (_loggedThisResult) return;
        if (widget.predictions.isEmpty) return;

        final sorted = [...widget.predictions]..sort((a, b) => b.finalProb.compareTo(a.finalProb));
        final top1 = sorted.first;

        // require ≥ 99% confidence
        final conf = top1.finalProb; // 0..1
        if (conf < 0.99) return;

        final user = FirebaseAuth.instance.currentUser;
        if (user == null) return;

        final nowUtc = DateTime.now().toUtc();
        final when = widget.takenAtUtc?.toUtc() ?? nowUtc;

        final data = <String, dynamic>{
            'userId': user.uid,
            'disease': top1.label,
            'confidence': conf,
            'lat': widget.lat,
            'lon': widget.lon,
            'stateCode': widget.stateCode,
            'timezone': widget.timezone,
            'date': Timestamp.fromDate(when),
            'createdAt': FieldValue.serverTimestamp(),
            'source': 'image_top1',
            'app': 'crop_care',
        };

        try {
            await FirebaseFirestore.instance.collection('disease_reports').add(data);
            _loggedThisResult = true;
            if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Logged high-confidence report (≥99%)')),
                );
            }
        } catch (e) {
            if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Failed to log report: $e')),
                );
            }
        }
    }

    Future<void> _runForecast() async {
        if (widget.lat == null || widget.lon == null) return;
        setState(() {
            _loadingForecast = true;
            _forecastJson = null;
            _nextWeekRiskForDetectedTop = null;
            _mgmtView = null;
        });

        try {
            final detected = _detectedTopDiseaseLabel();
            final resp = await LeafApi.predictForecast(
                lat: widget.lat!,
                lon: widget.lon!,
                timezone: widget.timezone,
                stateCode: widget.stateCode,
                forecastDays: 16,
                focusLabel: detected,   // <— add this param in your client
            );
            List<_DayRisk>? nextWeek;
            if (detected != null) {
                nextWeek = _extractNext7DayRisk(resp, detected);
            }

            // Build management view using peak risk for top when available
            _MgmtView? mgmtView;
            final mgmt = resp['management'] as Map<String, dynamic>?;
            if (mgmt != null) {
                final peak = resp['peak_risk_for_top'] as Map<String, dynamic>?;
                final riskStr =
                (peak?['risk']?.toString() ?? mgmt['risk']?.toString() ?? 'low').toLowerCase();
                final riskDateStr = peak?['date']?.toString();
                DateTime? riskDate;
                if (riskDateStr != null) {
                    try {
                        riskDate = DateTime.parse(riskDateStr).toLocal();
                    } catch (_) {}
                }
                mgmtView = _MgmtView(
                    disease: mgmt['for']?.toString() ?? detected ?? 'Top disease',
                    risk: riskStr,
                    riskDate: riskDate,
                    tips: List<String>.from(mgmt['tips'] ?? const []),
                    safety: Map<String, dynamic>.from(mgmt['safety'] ?? const {}),
                );
            }

            if (!mounted) return;
            setState(() {
                _loadingForecast = false;
                _forecastJson = resp;
                _nextWeekRiskForDetectedTop = nextWeek ?? const [];
                _mgmtView = mgmtView;
            });
        } catch (e) {
            if (!mounted) return;
            setState(() => _loadingForecast = false);
            ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Forecast prediction failed: $e')),
            );
        }
    }

    // ---------- Helpers ----------

    String? _detectedTopDiseaseLabel() {
        if (widget.predictions.isEmpty) return null;
        final sorted = [...widget.predictions]..sort((a, b) => b.finalProb.compareTo(a.finalProb));
        return sorted.first.label;
    }

    // Safe case-insensitive value reader for Map<dynamic,dynamic>
    String? _getCaseInsensitiveMapValue(Map map, String key) {
        final k = key.toLowerCase();
        for (final entry in map.entries) {
            final ek = entry.key?.toString().toLowerCase();
            if (ek == k) return entry.value?.toString();
        }
        return null;
    }

    List<_DayRisk> _extractNext7DayRisk(Map<String, dynamic> resp, String diseaseLabel) {
        final now = DateTime.now().toUtc();
        final cutoff = now.add(const Duration(days: 7));
        final results = <_DayRisk>[];

        // Preferred: resp["daily"] = [{ "date": "YYYY-MM-DD", "risk_map": {"Label":"high"}}, ...]
        final daily = resp['daily'];
        if (daily is List) {
            for (final d in daily.take(14)) {
                if (d is! Map<String, dynamic>) continue;

                final dateStr = d['date']?.toString();
                if (dateStr == null) continue;

                DateTime dt;
                try {
                    dt = DateTime.parse(dateStr).toUtc();
                } catch (_) {
                    continue;
                }
                if (dt.isBefore(now) || dt.isAfter(cutoff)) continue;

                String? riskForLabel;

                // Prefer risk_map (Map)
                final riskMap = d['risk_map'];
                if (riskMap is Map) {
                    final v = _getCaseInsensitiveMapValue(riskMap, diseaseLabel);
                    if (v != null) riskForLabel = v.toLowerCase();
                }

                // Fallback: risks as list of {label,risk}
                if (riskForLabel == null) {
                    final risksAny = d['risks'];
                    if (risksAny is List) {
                        for (final r in risksAny) {
                            if (r is Map<String, dynamic>) {
                                final lbl = r['label']?.toString();
                                if (lbl != null && lbl.toLowerCase() == diseaseLabel.toLowerCase()) {
                                    final v = r['risk']?.toString();
                                    if (v != null) {
                                        riskForLabel = v.toLowerCase();
                                        break;
                                    }
                                }
                            }
                        }
                    } else if (risksAny is Map) {
                        final v = _getCaseInsensitiveMapValue(risksAny, diseaseLabel);
                        if (v != null) riskForLabel = v.toLowerCase();
                    }
                }

                if (riskForLabel != null) {
                    results.add(_DayRisk(dt, _normalizeRisk(riskForLabel)));
                }
            }
        }

        // Fallback: resp["risks_by_disease"]["<label>"] = [{"date":"...","risk":"..."}]
        if (results.isEmpty) {
            final byDis = resp['risks_by_disease'];
            if (byDis is Map && byDis[diseaseLabel] is List) {
                final lst = byDis[diseaseLabel] as List;
                for (final r in lst) {
                    if (r is! Map) continue;
                    final dateStr = r['date']?.toString();
                    if (dateStr == null) continue;

                    DateTime dt;
                    try {
                        dt = DateTime.parse(dateStr).toUtc();
                    } catch (_) {
                        continue;
                    }
                    if (dt.isBefore(now) || dt.isAfter(cutoff)) continue;

                    final risk = r['risk']?.toString().toLowerCase();
                    if (risk != null) results.add(_DayRisk(dt, _normalizeRisk(risk)));
                }
            }
        }

        // Last resort: single overall risk via management block
        if (results.isEmpty) {
            final mgmt = resp['management'];
            if (mgmt is Map && (mgmt['for']?.toString().toLowerCase() == diseaseLabel.toLowerCase())) {
                final risk = mgmt['risk']?.toString();
                if (risk != null) results.add(_DayRisk(now, _normalizeRisk(risk)));
            }
        }

        results.sort((a, b) => a.date.compareTo(b.date));
        return results.take(7).toList();
    }

    String _normalizeRisk(String r) {
        final s = r.toLowerCase();
        if (s.contains('high')) return 'high';
        if (s.contains('med')) return 'medium';
        return 'low';
    }
}

// ---------- UI pieces ----------

class _SectionHeader extends StatelessWidget {
    final String text;
    const _SectionHeader(this.text);

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
                text,
                style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.2,
                ),
            ),
        );
    }
}

class _PrimaryButton extends StatelessWidget {
    final Widget icon;
    final String label;
    final VoidCallback? onPressed;
    const _PrimaryButton({
        required this.icon,
        required this.label,
        required this.onPressed,
    });

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        return SizedBox(
            height: 48,
            child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                    elevation: 0,
                    backgroundColor: theme.colorScheme.primary,
                    foregroundColor: theme.colorScheme.onPrimary,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                ),
                onPressed: onPressed,
                icon: icon,
                label: Text(
                    label,
                    style: theme.textTheme.labelLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.2,
                        color: theme.colorScheme.onPrimary,
                    ),
                ),
            ),
        );
    }
}

/// Coverage status (neutral/primary only, no green boxes)
class _CoverageStatus extends StatelessWidget {
    const _CoverageStatus({required this.coverage});
    final double coverage; // 0..1

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        final pct = (coverage.clamp(0, 1) * 100).toStringAsFixed(1);

        final isHigh = coverage > 0.30;
        final isMed = coverage > 0.15;
        final color = isHigh
            ? Colors.red
            : isMed
            ? Colors.amber
            : theme.colorScheme.primary;

        return Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
                color: theme.colorScheme.surfaceVariant.withOpacity(0.25),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: theme.colorScheme.outlineVariant.withOpacity(0.5)),
            ),
            child: Row(
                children: [
                    Icon(Icons.blur_circular, size: 24, color: color),
                    const SizedBox(width: 12),
                    Expanded(
                        child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                                Text('Leaf area affected',
                                    style: theme.textTheme.titleSmall?.copyWith(
                                        color: theme.textTheme.bodySmall?.color?.withOpacity(0.75),
                                    )),
                                const SizedBox(height: 8),
                                ClipRRect(
                                    borderRadius: BorderRadius.circular(6),
                                    child: LinearProgressIndicator(
                                        value: coverage.clamp(0, 1),
                                        minHeight: 8,
                                        backgroundColor: theme.colorScheme.surfaceVariant.withOpacity(0.6),
                                        valueColor: AlwaysStoppedAnimation<Color>(color),
                                    ),
                                ),
                            ],
                        ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                        '$pct%',
                        style: theme.textTheme.headlineSmall?.copyWith(
                            fontFeatures: const [FontFeature.tabularFigures()],
                            color: color,
                            fontWeight: FontWeight.w800,
                        ),
                    ),
                ],
            ),
        );
    }
}

/// #1 prediction (emphasized, no green accents)
class _TopPredictionCard extends StatelessWidget {
    final TopPrediction p;
    const _TopPredictionCard({required this.p});

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        final pct = (p.finalProb.clamp(0, 1) * 100).toStringAsFixed(1);
        final accent = theme.colorScheme.primary;

        return Card(
            color: Colors.white,
            elevation: 3,
            margin: const EdgeInsets.only(bottom: 16),
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
                side: BorderSide(color: theme.colorScheme.outlineVariant),
            ),
            child: Padding(
                padding: const EdgeInsets.all(18),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                        // Top row uses Wrap to avoid overflow on small widths
                        Wrap(
                            alignment: WrapAlignment.spaceBetween,
                            runSpacing: 8,
                            children: [
                                Text('Confidence',
                                    style: theme.textTheme.titleSmall
                                        ?.copyWith(color: theme.colorScheme.onSurface.withOpacity(0.70))),
                                Text(
                                    '$pct%',
                                    style: theme.textTheme.headlineMedium?.copyWith(
                                        color: accent,
                                        fontWeight: FontWeight.w900,
                                        fontFeatures: const [FontFeature.tabularFigures()],
                                    ),
                                ),
                            ],
                        ),
                        const SizedBox(height: 10),
                        const Divider(height: 24),
                        Text('Primary diagnosis', style: theme.textTheme.bodySmall),
                        const SizedBox(height: 4),
                        Text(
                            p.label,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
                        ),
                    ],
                ),
            ),
        );
    }
}

/// Ranks 2 & 3 prediction rows (neutral style, no green)
class _PredictionRow extends StatelessWidget {
    const _PredictionRow({required this.p});
    final TopPrediction p;

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        final pct = (p.finalProb.clamp(0, 1) * 100).toStringAsFixed(1);
        final light = theme.textTheme.bodyMedium?.color?.withOpacity(0.72);

        return Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
                children: [
                    SizedBox(
                        width: 30,
                        child: Text(
                            '${p.rank}.',
                            textAlign: TextAlign.right,
                            style: theme.textTheme.titleMedium?.copyWith(color: light, fontWeight: FontWeight.w600),
                        ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                        child: Text(
                            p.label,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: theme.textTheme.titleMedium,
                        ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                        '$pct%',
                        style: theme.textTheme.titleMedium?.copyWith(
                            color: light,
                            fontFeatures: const [FontFeature.tabularFigures()],
                        ),
                    ),
                ],
            ),
        );
    }
}

// === Next-week risk ===

class _DayRisk {
    final DateTime date;
    final String risk; // "low" | "medium" | "high"
    _DayRisk(this.date, this.risk);
}

class _NextWeekRiskCard extends StatelessWidget {
    final String diseaseLabel;
    final List<_DayRisk> days;
    const _NextWeekRiskCard({required this.diseaseLabel, required this.days});

    Color _riskColor(BuildContext context, String risk) {
        switch (risk) {
            case 'high':
                return Colors.red;
            case 'medium':
                return Colors.amber;
            default:
                return Theme.of(context).colorScheme.primary;
        }
    }

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);

        int peak = 0; // 0=low,1=medium,2=high
        for (final d in days) {
            final v = d.risk == 'high' ? 2 : d.risk == 'medium' ? 1 : 0;
            if (v > peak) peak = v;
        }
        final peakStr = peak == 2 ? 'HIGH' : peak == 1 ? 'MEDIUM' : 'LOW';
        final peakColor = _riskColor(context, peak == 2 ? 'high' : peak == 1 ? 'medium' : 'low');

        return Card(
            elevation: 0,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    _CardTitle(icon: Icons.trending_up, title: 'Next 7 days risk'),
                    const SizedBox(height: 6),
                    Text(diseaseLabel, style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 10),
                    // Use Wrap so the badge can flow to next line on narrow screens
                    Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                            _ChipBadge(label: 'PEAK: $peakStr', color: peakColor),
                        ],
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: days.map((d) {
                            final label = '${d.date.month}/${d.date.day}';
                            final c = _riskColor(context, d.risk);
                            return Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                                decoration: BoxDecoration(
                                    color: c.withOpacity(0.08),
                                    borderRadius: BorderRadius.circular(10),
                                    border: Border.all(color: c.withOpacity(0.45)),
                                ),
                                child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                        const Icon(Icons.calendar_today, size: 14),
                                        const SizedBox(width: 6),
                                        Text('$label • ${d.risk.toUpperCase()}',
                                            style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600)),
                                    ],
                                ),
                            );
                        }).toList(),
                    ),
                ]),
            ),
        );
    }
}

// === Management tips (no green accents) ===

class _MgmtView {
    final String disease;
    final String risk; // "low" | "medium" | "high"
    final DateTime? riskDate; // peak date if available
    final List<String> tips;
    final Map<String, dynamic> safety;
    _MgmtView({
        required this.disease,
        required this.risk,
        required this.riskDate,
        required this.tips,
        required this.safety,
    });
}

class _ManagementCard extends StatelessWidget {
    final _MgmtView view;
    const _ManagementCard({required this.view});

    Color _riskColor(String risk) {
        switch (risk) {
            case 'high':
                return Colors.red;
            case 'medium':
                return Colors.amber;
            default:
                return Colors.blueGrey; // neutral for low
        }
    }

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        final c = _riskColor(view.risk);
        final dateStr = view.riskDate != null ? '${view.riskDate!.month}/${view.riskDate!.day}' : null;

        return Card(
            elevation: 0,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    _CardTitle(icon: Icons.handyman_outlined, title: 'Management'),
                    const SizedBox(height: 8),
                    // Wrap here avoids horizontal overflow on small screens
                    Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: [
                            ConstrainedBox(
                                constraints: const BoxConstraints(minWidth: 100, maxWidth: 400),
                                child: Text(
                                    view.disease,
                                    maxLines: 2,
                                    overflow: TextOverflow.ellipsis,
                                    style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                                ),
                            ),
                            _ChipBadge(label: '${view.risk.toUpperCase()}${dateStr != null ? ' • $dateStr' : ''}', color: c),
                        ],
                    ),
                    const SizedBox(height: 12),
                    if (view.tips.isEmpty)
                        Text('No tips available.', style: theme.textTheme.bodyMedium)
                    else
                        Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: view.tips.map((t) {
                                return Padding(
                                    padding: const EdgeInsets.only(bottom: 10),
                                    child: Row(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                            const Padding(
                                                padding: EdgeInsets.only(top: 4),
                                                child: Icon(Icons.check_circle_outline, size: 18),
                                            ),
                                            const SizedBox(width: 8),
                                            Expanded(
                                                child: Text(t, style: theme.textTheme.bodyMedium?.copyWith(height: 1.25)),
                                            ),
                                        ],
                                    ),
                                );
                            }).toList(),
                        ),
                    if (view.safety.isNotEmpty) ...[
                        const SizedBox(height: 6),
                        const Divider(height: 24),
                        Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                                if (view.safety['disclaimer'] != null)
                                    _SafetyPill(icon: Icons.info_outline, text: view.safety['disclaimer'].toString()),
                                if (view.safety['ppe'] is List && (view.safety['ppe'] as List).isNotEmpty)
                                    _SafetyPill(icon: Icons.health_and_safety_outlined, text: 'PPE: ${(view.safety['ppe'] as List).join(", ")}'),
                                if (view.safety['bee_protection'] != null)
                                    _SafetyPill(icon: Icons.bug_report_outlined, text: view.safety['bee_protection'].toString()),
                            ],
                        ),
                    ],
                ]),
            ),
        );
    }
}

class _SafetyPill extends StatelessWidget {
    final IconData icon;
    final String text;
    const _SafetyPill({required this.icon, required this.text});

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        final border = theme.colorScheme.outlineVariant;
        return Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: border),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(icon, size: 16),
                const SizedBox(width: 6),
                ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 260),
                    child: Text(text, style: theme.textTheme.bodySmall?.copyWith(height: 1.2), overflow: TextOverflow.ellipsis),
                ),
            ]),
        );
    }
}

class _ChipBadge extends StatelessWidget {
    final String label;
    final Color color;
    const _ChipBadge({required this.label, required this.color});

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        return Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
                color: color.withOpacity(0.10),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: color.withOpacity(0.45)),
            ),
            child: Text(
                label,
                overflow: TextOverflow.fade,
                softWrap: false,
                style: theme.textTheme.labelLarge?.copyWith(
                    color: theme.colorScheme.onSurface,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.4,
                ),
            ),
        );
    }
}

class _CardTitle extends StatelessWidget {
    final IconData icon;
    final String title;
    const _CardTitle({required this.icon, required this.title});

    @override
    Widget build(BuildContext context) {
        final theme = Theme.of(context);
        return Row(
            children: [
                Container(
                    decoration: BoxDecoration(
                        color: theme.colorScheme.secondaryContainer,
                        borderRadius: BorderRadius.circular(10),
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(icon, size: 18, color: theme.colorScheme.onSecondaryContainer),
                ),
                const SizedBox(width: 8),
                Expanded(
                    child: Text(
                        title,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style:
                        theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700, letterSpacing: 0.2),
                    ),
                ),
            ],
        );
    }
}