import 'package:flutter/material.dart';

class WeatherPredictionScreen extends StatefulWidget {
  final double lat;
  final double lon;
  final String timezone;
  final String? stateCode;
  final String? focusLabel;
  final Map<String, dynamic>? forecastData; // pre-fetched response

  const WeatherPredictionScreen({
    super.key,
    required this.lat,
    required this.lon,
    required this.timezone,
    this.stateCode,
    this.focusLabel,
    this.forecastData,
  });

  @override
  State<WeatherPredictionScreen> createState() => _WeatherPredictionScreenState();
}

class _WeatherPredictionScreenState extends State<WeatherPredictionScreen> {
  /// UI window: 7 (default) or 14 — toggles locally with no network call
  int _window = 7;

  /// Full response map passed from ResultsScreen
  late final Map<String, dynamic>? _resp;

  @override
  void initState() {
    super.initState();
    _resp = widget.forecastData;
  }

  /// Selected window block (typed as Map<String, dynamic>)
  Map<String, dynamic> get _sel {
    final byWindow = _resp?['by_window'];
    if (byWindow is Map) {
      final block = byWindow[_window.toString()];
      if (block is Map) return Map<String, dynamic>.from(block);
    }
    // Fallback: try legacy fields if by_window missing
    if (_resp != null && (_resp!.containsKey('top') || _resp!.containsKey('management'))) {
      return {
        'top': _resp!['top'],
        'risk_summary': _resp!['risk_summary'],
        'daily': _resp!['daily'],
        'risks_by_disease': _resp!['risks_by_disease'],
        'peak_risk_by_disease': _resp!['peak_risk_by_disease'],
        'peak_risk_for_top': _resp!['peak_risk_for_top'],
        'peak_risk_for_focus': _resp!['peak_risk_for_focus'],
        'focus': _resp!['focus'],
        'management': _resp!['management'],
      };
    }
    return {};
  }

  /// Prefer the label from management block (exact backend match),
  /// else the passed focusLabel, else the top-1 label, else a generic title.
  String _displayLabel() {
    final mg = (_sel['management'] as Map?) ?? const {};
    final mgFor = mg['for']?.toString();
    if (mgFor != null && mgFor.trim().isNotEmpty) return mgFor;

    final navFocus = (widget.focusLabel ?? '').trim();
    if (navFocus.isNotEmpty) return navFocus;

    final top = _sel['top'] as List?;
    if (top != null && top.isNotEmpty) return (top.first as Map)['label']?.toString() ?? 'Disease';

    return 'Disease';
  }

  /// Overall risk for the chosen disease; falls back to summary.
  String _overallRiskForChosen() {
    final label = _displayLabel().toLowerCase();
    final top = _sel['top'] as List? ?? const [];
    for (final t in top) {
      if (t is Map) {
        final tl = t['label']?.toString().toLowerCase();
        if (tl == label) {
          final r = t['risk']?.toString();
          if (r != null && r.isNotEmpty) return r;
        }
      }
    }
    return ((_sel['risk_summary'] as Map?)?['overall']?.toString()) ?? 'low';
  }

  /// Peak date: prefer focus’ peak, else top’s peak. Return formatted MM/DD/YYYY (local).
  String? _peakDateStr() {
    Map? pick = (_sel['peak_risk_for_focus'] as Map?) ?? (_sel['peak_risk_for_top'] as Map?);
    final raw = pick?['date']?.toString();
    if (raw == null || raw.isEmpty) return null;
    try {
      // Parse ISO yyyy-mm-dd or yyyy-mm-ddTHH:mm...
      final dt = DateTime.parse(raw).toLocal();
      final mm = dt.month.toString().padLeft(2, '0');
      final dd = dt.day.toString().padLeft(2, '0');
      final yyyy = dt.year.toString();
      return '$mm/$dd/$yyyy';
    } catch (_) {
      return raw; // show raw if unknown format
    }
  }

  /// Tips from management block (if present)
  List<String> _tips() {
    final mg = (_sel['management'] as Map?) ?? const {};
    return (mg['tips'] as List?)?.map((e) => e.toString()).toList() ?? const <String>[];
  }

  Color _riskColor(BuildContext context, String risk) {
    final r = risk.toLowerCase();
    if (r.startsWith('high')) return Colors.red;
    if (r.startsWith('med')) return Colors.amber;
    return Theme.of(context).colorScheme.primary;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final hasData = _resp != null &&
        ((_resp!['by_window'] is Map && (_resp!['by_window'] as Map).isNotEmpty) ||
            _resp!.containsKey('top') ||
            _resp!.containsKey('management'));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Forecast Risk Overview'),
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: !hasData
            ? Center(
          child: Text(
            'No forecast data provided.',
            style: theme.textTheme.bodyLarge,
          ),
        )
            : Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Window dropdown (7/14)
            Row(
              children: [
                const Icon(Icons.timelapse, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: DropdownButtonFormField<int>(
                    value: _window,
                    decoration: const InputDecoration(
                      isDense: true,
                      border: OutlineInputBorder(),
                      labelText: 'Forecast window',
                    ),
                    items: const [
                      DropdownMenuItem(value: 7, child: Text('Next 7 days')),
                      DropdownMenuItem(value: 14, child: Text('Next 14 days')),
                    ],
                    onChanged: (v) {
                      if (v == null) return;
                      setState(() => _window = v);
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            _InfoCard(
              title: _displayLabel(),
              risk: _overallRiskForChosen(),
              peakDate: _peakDateStr(),
              tips: _tips(),
              colorForRisk: _riskColor,
            ),
          ],
        ),
      ),
    );
  }
}

// ---------- UI sub-widgets ----------

class _InfoCard extends StatelessWidget {
  final String title;
  final String risk;
  final String? peakDate;
  final List<String> tips;
  final Color Function(BuildContext, String) colorForRisk;

  const _InfoCard({
    required this.title,
    required this.risk,
    required this.peakDate,
    required this.tips,
    required this.colorForRisk,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final c = colorForRisk(context, risk);

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(
            children: [
              Container(
                decoration: BoxDecoration(
                  color: theme.colorScheme.secondaryContainer,
                  borderRadius: BorderRadius.circular(10),
                ),
                padding: const EdgeInsets.all(8),
                child: Icon(Icons.eco_outlined, size: 18, color: theme.colorScheme.onSecondaryContainer),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),

          // Risk + Peak date badges
          Wrap(
            spacing: 8,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              _ChipBadge(label: risk.toUpperCase(), color: c),
              if (peakDate != null)
                _ChipBadge(label: 'Peak: $peakDate', color: theme.colorScheme.outline),
            ],
          ),

          const SizedBox(height: 14),

          // Management tips list
          if (tips.isEmpty)
            Text('No management tips available.', style: theme.textTheme.bodyMedium)
          else
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: tips.take(6).map((t) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.only(top: 3),
                        child: Icon(Icons.check_circle_outline, size: 18),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          t,
                          style: theme.textTheme.bodyMedium?.copyWith(height: 1.25),
                        ),
                      ),
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
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.45)),
      ),
      child: Text(
        label,
        overflow: TextOverflow.fade,
        softWrap: false,
        style: theme.textTheme.labelLarge?.copyWith(
          fontWeight: FontWeight.w700,
          letterSpacing: 0.3,
        ),
      ),
    );
  }
}