// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

// Localization
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:crop_care/core/localization/locale_controller.dart';
import 'package:crop_care/l10n/app_localizations.dart';

// Theming + entry screen
import 'package:crop_care/theme/theme.dart';
import 'package:crop_care/screens/login_screen.dart';

// Firebase
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';

// FIPS service + your existing county loader/types
import 'package:crop_care/services/fips_service.dart';
import 'package:crop_care/widgets/disease_map.dart'
    show loadCountiesFromAsset, CountyPolygon;

Future<void> _initFips() async {
  try {
    // Make sure this asset is listed in pubspec.yaml
    // flutter:
    //   assets:
    //     - assets/geo/us_counties_simplified.json
    final List<CountyPolygon> counties =
    await loadCountiesFromAsset('assets/geo/us_counties_simplified.json');

    FipsService.init(counties);
    debugPrint('[FIPS] Loaded ${counties.length} county polygons');
  } catch (e, st) {
    debugPrint('[FIPS] init failed: $e');
    debugPrintStack(stackTrace: st);
    // App will still run; saves depending on FIPS will be skipped if init fails.
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  // Locale
  await LocaleController.instance.init();

  // Firebase
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  // FIPS polygons/index (load once at app start)
  await _initFips();

  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  void initState() {
    super.initState();
    LocaleController.instance.addListener(_onLangChanged);
  }

  @override
  void dispose() {
    LocaleController.instance.removeListener(_onLangChanged);
    super.dispose();
  }

  void _onLangChanged() => setState(() {});

  @override
  Widget build(BuildContext context) {
    final locale = LocaleController.instance.locale;

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      onGenerateTitle: (ctx) => AppLocalizations.of(ctx)!.appTitle,
      locale: locale,
      supportedLocales: const [Locale('en'), Locale('es')],
      localizationsDelegates: const [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      themeMode: ThemeMode.system,
      theme: CropCareTheme.lightTheme,
      darkTheme: CropCareTheme.darkTheme,
      home: const LoginScreen(),
    );
  }
}