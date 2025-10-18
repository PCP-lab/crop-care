import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'home_screen.dart';

const kOnboardingSeenKey = 'onboarding_seen_v1';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _controller = CarouselSliderController();
  int _currentIndex = 0;

  final List<Map<String, String>> _pages = [
    {
      "image": "assets/onboarding/images/onboarding_image1.png",
      "title": "STEP 1",
      "description":
      "Identify plant diseases in seconds. Upload a photo, describe what you see, and get instant insights to protect your crops."
    },
    {
      "image": "assets/onboarding/images/onboarding_image2.png",
      "title": "STEP 2",
      "description":
      "Review personalized solutions tailored to your crop type and region."
    },
    {
      "image": "assets/onboarding/images/onboarding_image3.png",
      "title": "STEP 3",
      "description":
      "Track crop health over time and take preventive measures for a better harvest."
    },
  ];

  Future<void> _completeAndGoHome() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(kOnboardingSeenKey, true);
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const HomeScreen()),
          (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      backgroundColor: Theme.of(context).brightness == Brightness.dark
          ? Colors.black
          : Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            // Skip
            Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: TextButton(
                  onPressed: _completeAndGoHome,
                  child: Text(
                    "Skip",
                    style: textTheme.bodyMedium?.copyWith(
                      color: cs.primary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),

            // Carousel
            Expanded(
              child: CarouselSlider.builder(
                carouselController: _controller,
                itemCount: _pages.length,
                itemBuilder: (context, index, realIndex) {
                  final page = _pages[index];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        // Image
                        Flexible(
                          flex: 10, // gives more height to image
                          child: Padding(
                            padding: const EdgeInsets.symmetric(vertical: 24),
                            child: Image.asset(
                              page["image"]!,
                              fit: BoxFit.contain,
                            ),
                          ),
                        ),

                        const SizedBox(height: 12),

                        // Title (bold, theme aware)
                        Text(
                          page["title"]!,
                          textAlign: TextAlign.center,
                          style: textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),

                        const SizedBox(height: 8),

                        // Description (body text, theme aware)
                        Flexible(
                          fit: FlexFit.loose,
                          child: Text(
                            page["description"]!,
                            textAlign: TextAlign.center,
                            style: textTheme.bodyMedium,
                          ),
                        ),
                      ],
                    ),
                  );
                },
                options: CarouselOptions(
                  viewportFraction: 1,
                  enableInfiniteScroll: false,
                  onPageChanged: (index, reason) =>
                      setState(() => _currentIndex = index),
                ),
              ),
            ),

            // Dots
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                _pages.length,
                    (index) => AnimatedContainer(
                  duration: const Duration(milliseconds: 250),
                  margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 16),
                  height: 8,
                  width: _currentIndex == index ? 20 : 8,
                  decoration: BoxDecoration(
                    color: _currentIndex == index
                        ? cs.primary
                        : cs.primary.withOpacity(0.3),
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
            ),

            // Next / Get Started
            Padding(
              padding: const EdgeInsets.all(16),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () async {
                    final last = _currentIndex == _pages.length - 1;
                    if (last) {
                      await _completeAndGoHome();
                    } else {
                      _controller.nextPage(
                        duration: const Duration(milliseconds: 300),
                        curve: Curves.easeOut,
                      );
                    }
                  },
                  child: Text(
                    _currentIndex == _pages.length - 1
                        ? "Get Started"
                        : "Next",
                    style: textTheme.labelLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: Theme.of(context)
                          .colorScheme
                          .onPrimary, // ensures readable on primary button
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}