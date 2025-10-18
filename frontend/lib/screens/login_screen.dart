import 'dart:async';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import 'home_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _auth = FirebaseAuth.instance;

  final _phoneCtrl = TextEditingController(); // E.164 format: +1..., +91..., etc.
  final _codeCtrl  = TextEditingController();

  String? _verificationId;
  int? _resendToken;

  bool _sending = false;
  bool _verifying = false;
  String? _error;

  int _secondsLeft = 0;
  Timer? _timer;

  @override
  void dispose() {
    _phoneCtrl.dispose();
    _codeCtrl.dispose();
    _timer?.cancel();
    super.dispose();
  }

  void _startCooldown([int seconds = 60]) {
    _timer?.cancel();
    setState(() => _secondsLeft = seconds);
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (_secondsLeft <= 1) {
        t.cancel();
        setState(() => _secondsLeft = 0);
      } else {
        setState(() => _secondsLeft--);
      }
    });
  }

  Future<void> _sendCode() async {
    FocusScope.of(context).unfocus();
    setState(() { _sending = true; _error = null; });

    try {
      await _auth.verifyPhoneNumber(
        phoneNumber: _phoneCtrl.text.trim(),
        forceResendingToken: _resendToken,
        timeout: const Duration(seconds: 60),
        verificationCompleted: (PhoneAuthCredential cred) async {
          await _signInWithCredential(cred); // Android may auto-complete
        },
        verificationFailed: (FirebaseAuthException e) {
          setState(() => _error = e.message ?? e.code);
        },
        codeSent: (String verificationId, int? resendToken) {
          setState(() {
            _verificationId = verificationId;
            _resendToken = resendToken;
            _error = null;
          });
          _startCooldown();
        },
        codeAutoRetrievalTimeout: (String verificationId) {
          setState(() => _verificationId = verificationId);
        },
      );
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _confirmCode() async {
    if (_verificationId == null) {
      setState(() => _error = 'No verification in progress.');
      return;
    }
    setState(() { _verifying = true; _error = null; });
    try {
      final cred = PhoneAuthProvider.credential(
        verificationId: _verificationId!,
        smsCode: _codeCtrl.text.trim(),
      );
      await _signInWithCredential(cred);
    } on FirebaseAuthException catch (e) {
      setState(() => _error = e.message ?? e.code);
    } finally {
      if (mounted) setState(() => _verifying = false);
    }
  }

  Future<void> _signInWithCredential(PhoneAuthCredential cred) async {
    await _auth.signInWithCredential(cred); // signs in (creates user if new)
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Signed in successfully')),
    );

    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const HomeScreen()),
          (route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final isCodeStage = _verificationId != null;
    final theme = Theme.of(context);
    final onCard = theme.colorScheme.onSurface;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        title: const Text('Sign in'),
        centerTitle: true,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Background image
          Image.asset(
            'assets/home/images/homePage_image.jpg',
            fit: BoxFit.cover,
          ),
          // Gradient overlay for contrast
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.black.withOpacity(0.45),
                  Colors.black.withOpacity(0.25),
                ],
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
              ),
            ),
          ),
          // Centered content
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 420),
                  child: _FrostedCard(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        // App title / tagline
                        Text(
                          'CropCare',
                          style: theme.textTheme.headlineMedium?.copyWith(
                            fontWeight: FontWeight.w700,
                            color: onCard,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 6),
                        Text(
                          isCodeStage
                              ? 'Verify your phone number'
                              : 'Sign in with your phone to continue',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: onCard.withOpacity(0.9),
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 20),

                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 250),
                          switchInCurve: Curves.easeOut,
                          switchOutCurve: Curves.easeIn,
                          child: isCodeStage
                              ? _buildCodeStage(theme, onCard)
                              : _buildPhoneStage(theme, onCard),
                        ),

                        if (_error != null) ...[
                          const SizedBox(height: 14),
                          Row(
                            children: [
                              Icon(Icons.error_outline, color: Colors.red.shade400, size: 18),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  _error!,
                                  style: theme.textTheme.bodySmall?.copyWith(color: Colors.red.shade300),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPhoneStage(ThemeData theme, Color onCard) {
    return Column(
      key: const ValueKey('phoneStage'),
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _LabeledField(
          label: 'Phone number',
          child: TextField(
            controller: _phoneCtrl,
            keyboardType: TextInputType.phone,
            textInputAction: TextInputAction.done,
            decoration: _inputDecoration(hint: '+1 408 555 1234'),
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          height: 48,
          child: ElevatedButton(
            onPressed: _sending ? null : _sendCode,
            style: ElevatedButton.styleFrom(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              elevation: 0,
            ),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: _sending
                  ? const SizedBox(
                height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2.4),
              )
                  : const Text('Send code'),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCodeStage(ThemeData theme, Color onCard) {
    return Column(
      key: const ValueKey('codeStage'),
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: Text(
            'Code sent to ${_phoneCtrl.text}',
            style: theme.textTheme.bodySmall?.copyWith(color: onCard.withOpacity(0.8)),
          ),
        ),
        const SizedBox(height: 10),
        _LabeledField(
          label: '6-digit code',
          child: TextField(
            controller: _codeCtrl,
            keyboardType: TextInputType.number,
            maxLength: 6,
            decoration: _inputDecoration(hint: 'Enter code').copyWith(counterText: ''),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 48,
          child: ElevatedButton(
            onPressed: _verifying ? null : _confirmCode,
            style: ElevatedButton.styleFrom(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              elevation: 0,
            ),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 200),
              child: _verifying
                  ? const SizedBox(
                height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2.4),
              )
                  : const Text('Verify & continue'),
            ),
          ),
        ),
        const SizedBox(height: 6),
        TextButton(
          onPressed: (_secondsLeft == 0 && !_sending) ? _sendCode : null,
          child: Text(
            _secondsLeft == 0 ? 'Resend code' : 'Resend in $_secondsLeft s',
            style: TextStyle(color: onCard.withOpacity(0.9)),
          ),
        ),
      ],
    );
  }

  InputDecoration _inputDecoration({required String hint}) {
    final theme = Theme.of(context);
    return InputDecoration(
      hintText: hint,
      filled: true,
      fillColor: theme.colorScheme.surface.withOpacity(0.75),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: theme.colorScheme.outline.withOpacity(0.3)),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: theme.colorScheme.outline.withOpacity(0.25)),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: theme.colorScheme.primary.withOpacity(0.6), width: 1.6),
      ),
      hintStyle: TextStyle(color: theme.colorScheme.onSurface.withOpacity(0.55)),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
    );
  }
}

class _FrostedCard extends StatelessWidget {
  final Widget child;
  const _FrostedCard({required this.child});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 14, sigmaY: 14),
        child: Container(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 20),
          decoration: BoxDecoration(
            color: scheme.surface.withOpacity(0.55),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: scheme.outline.withOpacity(0.2)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.22),
                blurRadius: 24,
                offset: const Offset(0, 12),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );
  }
}

class _LabeledField extends StatelessWidget {
  final String label;
  final Widget child;
  const _LabeledField({required this.label, required this.child});

  @override
  Widget build(BuildContext context) {
    final onSurface = Theme.of(context).colorScheme.onSurface;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.labelLarge?.copyWith(
            color: onSurface.withOpacity(0.9),
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 8),
        child,
      ],
    );
  }
}