// lib/widgets/custom_app_bar.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../core/localization/app_language.dart';
import '../core/localization/locale_controller.dart';
import '../screens/login_screen.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final List<Widget>? extraActions;

  const CustomAppBar({
    super.key,
    required this.title,
    this.extraActions,
  });

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);

  Future<void> _handleLogout(BuildContext context) async {
    try {
      await FirebaseAuth.instance.signOut();
      if (context.mounted) {
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (_) => const LoginScreen()),
              (route) => false,
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Logout failed: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final current = LocaleController.instance.language;
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return AppBar(
      title: Text(title),
      actions: [
        if (extraActions != null) ...extraActions!,

        // Language dropdown (kept)
        Padding(
          padding: const EdgeInsets.only(right: 4),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<AppLanguage>(
              value: current,
              borderRadius: const BorderRadius.all(Radius.circular(12)),
              dropdownColor: Theme.of(context).brightness == Brightness.dark
                  ? Colors.black
                  : Colors.white,
              onChanged: (v) async {
                if (v != null) {
                  await LocaleController.instance.setLanguage(v); // persist + notify
                }
              },
              items: const [
                DropdownMenuItem(
                  value: AppLanguage.en,
                  child: Text('English'),
                ),
                DropdownMenuItem(
                  value: AppLanguage.es,
                  child: Text('Español'),
                ),
              ],
              style: textTheme.bodyMedium?.copyWith(
                color: scheme.onSurface, // ensure text is visible
              ),
              icon: Icon(Icons.language, color: scheme.onSurface),
            ),
          ),
        ),

        // Logout action (door/arrow icon)
        IconButton(
          tooltip: 'Logout',
          icon: Icon(Icons.logout, color: scheme.onSurface),
          onPressed: () => _handleLogout(context),
        ),
        const SizedBox(width: 8),
      ],
    );
  }
}