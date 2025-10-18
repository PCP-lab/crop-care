import 'dart:async';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'onboarding_screen.dart';
import 'home_screen.dart';
import 'package:crop_care/l10n/app_localizations.dart';

const _kOnboardingSeenKey = 'onboarding_seen_v1'; // bump to _v2 if you redesign

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _decideNext(); // run the shared pref check here
  }

  Future<void> _decideNext() async {
    // Optional: tiny delay to let splash/logo show smoothly
    await Future.delayed(const Duration(milliseconds: 400));

    final prefs = await SharedPreferences.getInstance();
    final seen = prefs.getBool(_kOnboardingSeenKey) ?? false;

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => seen ? const HomeScreen() : const OnboardingScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Color(0xFF819067),
      body: Center(
        child: _SplashArt(),
      ),
    );
  }
}

class _SplashArt extends StatelessWidget {
  const _SplashArt();

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Image.asset(
          "assets/onboarding/images/onboarding_image3.png",
          height: 200,
          fit: BoxFit.cover,
        ),
        const SizedBox(height: 16),
        Text(
          l10n.appTitle, // e.g., "Crop Care"
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 32),
        ),
        Text(
          l10n.splashSubtitle, // e.g., "Detect. Protect. Prosper."
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 20,
            fontStyle: FontStyle.italic,
          ),
        ),
      ],
    );
  }
}