// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Spanish Castilian (`es`).
class AppLocalizationsEs extends AppLocalizations {
  AppLocalizationsEs([String locale = 'es']) : super(locale);

  @override
  String get appTitle => 'Crop Care';

  @override
  String get splashSubtitle => 'Detectar. Proteger. Prosperar.';

  @override
  String get homeTitle => 'Inicio';

  @override
  String get homeQuestion => '¿Mis plantas están saludables?';

  @override
  String get homeInstruction => 'Toca la tarjeta para comenzar';

  @override
  String plantName(String name) {
    return '$name';
  }

  @override
  String plantFactMoisture(int percent) {
    return 'Nivel de humedad $percent%';
  }

  @override
  String get resultsTitle => 'Resultados';

  @override
  String get uploadPhoto => 'Subir la foto';

  @override
  String get camera => 'Cámara';

  @override
  String get gallery => 'Galería';

  @override
  String get description => 'Descripción';

  @override
  String get dateNoticed => 'Fecha observada';

  @override
  String get diagnose => 'Diagnosticar';

  @override
  String get descriptionHint => 'Describe lo que ves (manchas, decoloración, plagas, marchitez, etc.)';
}
