// lib/core/localization/locale_controller.dart
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'app_language.dart';

class LocaleController {
  LocaleController._();
  static final LocaleController instance = LocaleController._();

  static const _key = 'preferred_locale_code';

  AppLanguage _language = AppLanguage.en;
  AppLanguage get language => _language;
  Locale get locale => Locale(_language.code);

  final List<VoidCallback> _listeners = [];

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _language = AppLanguageX.fromCode(prefs.getString(_key));
  }

  Future<void> setLanguage(AppLanguage lang) async {
    if (_language == lang) return;
    _language = lang;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, lang.code);
    for (final l in List<VoidCallback>.from(_listeners)) {
      l();
    }
  }

  void addListener(VoidCallback listener) => _listeners.add(listener);
  void removeListener(VoidCallback listener) => _listeners.remove(listener);
}