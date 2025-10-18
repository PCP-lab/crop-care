import 'package:flutter/material.dart';
import 'package:crop_care/theme/text_theme.dart';

class _Palette {
  static const primary = Color(0xFFA0C878); // #A0C878
  static const scaffold = Color(0xFFFFFDF6); // #FFFDF6
  static const surface  = Color(0xFFFAF6E9); // #FAF6E9
  static const secondary = Color(0xFFDDEB9D); // #DDEB9D
  static const onPrimary = Colors.white;
  static const scaffoldDark = Color(0xFF121212); // Scaffold / background
  static const surfaceDark  = Color(0xFF1E1E1E); // Cards, dialogs
}

class CropCareTheme {
  CropCareTheme._();

  static ThemeData lightTheme = ThemeData(
    useMaterial3: true,
    fontFamily: 'Poppins',
    brightness: Brightness.light,
    primaryColor: _Palette.primary,
    scaffoldBackgroundColor: _Palette.onPrimary,
    appBarTheme: const AppBarTheme(
      backgroundColor: _Palette.primary, // sits on scaffold
      elevation: 0,
      centerTitle: true,
      foregroundColor: Colors.black,       // title/icons in light
      titleTextStyle: TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.w700,
        color: Colors.black,
      ),
      iconTheme: IconThemeData(color: Colors.black),
      actionsIconTheme: IconThemeData(color: Colors.black),
    ),

    colorScheme: const ColorScheme(
      brightness: Brightness.light,
      primary: _Palette.primary,
      onPrimary: _Palette.onPrimary,
      secondary: _Palette.secondary,
      onSecondary: Colors.black87,
      surface: _Palette.surface,
      onSurface: Colors.black87,
      background: _Palette.scaffold,
      onBackground: Colors.black87,
      error: Colors.red,
      onError: Colors.white,
    ),

    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: _Palette.primary,
        foregroundColor: _Palette.onPrimary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),

    // in CropCareTheme
    cardTheme: const CardThemeData(
      color: _Palette.primary,                // light: FA F6E9
      elevation: 0,
      margin: EdgeInsets.all(12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(12)),
      ),
      surfaceTintColor: Colors.transparent,   // <- important for M3
    ),

    // ✅ Black text for light theme
    textTheme: CropCareTextTheme.lightTextTheme,
  );

  static ThemeData darkTheme = ThemeData(
    useMaterial3: true,
    fontFamily: 'Poppins',
    brightness: Brightness.dark,
    primaryColor: _Palette.primary,
    scaffoldBackgroundColor: _Palette.scaffoldDark.withOpacity(0.2),

    appBarTheme: const AppBarTheme(
      backgroundColor: _Palette.primary,
      elevation: 0,
      centerTitle: true,
      foregroundColor: Colors.white,       // title/icons in dark
      titleTextStyle: TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.w700,
        color: Colors.white,
      ),
      iconTheme: IconThemeData(color: Colors.white),
      actionsIconTheme: IconThemeData(color: Colors.white),
    ),

    colorScheme: const ColorScheme(
      brightness: Brightness.dark,
      primary: _Palette.primary,
      onPrimary: _Palette.onPrimary,
      secondary: _Palette.secondary,
      onSecondary: Colors.white,
      surface: _Palette.surface,
      onSurface: Colors.white,
      background: _Palette.scaffold,
      onBackground: Colors.white,
      error: Colors.redAccent,
      onError: Colors.white,
    ),


    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: _Palette.primary,
        foregroundColor: _Palette.onPrimary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),

    cardTheme: const CardThemeData(
      color: _Palette.primary,
      elevation: 0,
      margin: EdgeInsets.all(12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(12)),
      ),
      surfaceTintColor: Colors.transparent,
    ),

    // ✅ White text for dark theme
    textTheme: CropCareTextTheme.darkTextTheme,
  );
}