// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'Crop Care';

  @override
  String get splashSubtitle => 'Detect. Protect. Prosper.';

  @override
  String get homeTitle => 'Home';

  @override
  String get homeQuestion => 'Are my plants healthy?';

  @override
  String get homeInstruction => 'Tap on the card to start';

  @override
  String plantName(String name) {
    return '$name';
  }

  @override
  String plantFactMoisture(int percent) {
    return 'Moisture level $percent%';
  }

  @override
  String get resultsTitle => 'Results';

  @override
  String get uploadPhoto => 'Upload the photo';

  @override
  String get camera => 'Camera';

  @override
  String get gallery => 'Gallery';

  @override
  String get description => 'Description';

  @override
  String get dateNoticed => 'Date noticed';

  @override
  String get diagnose => 'Diagnose';

  @override
  String get descriptionHint => 'Describe what you see(spots,discoloration,pests,wilting, ec.)';
}
