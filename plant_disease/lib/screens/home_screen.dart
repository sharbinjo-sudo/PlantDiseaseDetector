import 'dart:io';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:flutter/services.dart'; // 👈 for SystemNavigator.pop()
import '../widgets/navbar.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  DateTime? _lastExitAttempt;

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: _handleAppExit,
      child: Scaffold(
        backgroundColor: const Color(0xFFF6FAF7),
        body: SingleChildScrollView(
          child: Column(
            children: [
              const NavBar(activePage: "Home"),
              const SizedBox(height: 40),
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 40, vertical: 30),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    bool isSmallScreen = constraints.maxWidth < 900;
                    return isSmallScreen
                        ? Column(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              _buildTextSection(context, isSmallScreen),
                              const SizedBox(height: 40),
                              _buildImageSection(isSmallScreen),
                            ],
                          )
                        : Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Expanded(
                                flex: 5,
                                child:
                                    _buildTextSection(context, isSmallScreen),
                              ),
                              const SizedBox(width: 40),
                              Expanded(
                                flex: 5,
                                child: _buildImageSection(isSmallScreen),
                              ),
                            ],
                          );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ✅ Android back-press exit confirmation
  Future<bool> _handleAppExit() async {
    final now = DateTime.now();

    if (_lastExitAttempt == null ||
        now.difference(_lastExitAttempt!) > const Duration(seconds: 2)) {
      _lastExitAttempt = now;
      Fluttertoast.showToast(
        msg: "Press back again to exit",
        toastLength: Toast.LENGTH_SHORT,
        gravity: ToastGravity.BOTTOM,
        backgroundColor: Colors.black87,
        textColor: Colors.white,
      );
      return false;
    }

    if (Platform.isAndroid) {
      SystemNavigator.pop(); // Clean Android exit
      return false;
    } else {
      return true;
    }
  }

  // 🌿 Text Section
  Widget _buildTextSection(BuildContext context, bool isSmallScreen) {
    return Column(
      crossAxisAlignment:
          isSmallScreen ? CrossAxisAlignment.center : CrossAxisAlignment.start,
      children: [
        Text(
          "Plant Leaf Disease",
          textAlign: isSmallScreen ? TextAlign.center : TextAlign.start,
          style: const TextStyle(
            fontSize: 46,
            fontWeight: FontWeight.bold,
            color: Color(0xFF0c4128),
          ),
        ),
        const SizedBox(height: 16),
        Text(
          "Find diseases in your plants instantly by uploading a photo of the leaf. "
          "Powered by Deep Learning.",
          textAlign: isSmallScreen ? TextAlign.center : TextAlign.start,
          style: const TextStyle(
            fontSize: 18,
            color: Colors.black54,
            height: 1.5,
          ),
        ),
        const SizedBox(height: 30),
        Center(
          child: SizedBox(
            height: 56,
            child: ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/predict');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF28a745),
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 0),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(50),
                ),
                elevation: 3,
              ),
              child: const Text(
                "Get Started",
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // 🌿 Image Section
  Widget _buildImageSection(bool isSmallScreen) {
    return Center(
      child: Image.asset(
        'assets/hero-img.png',
        height: isSmallScreen ? 280 : 400,
        fit: BoxFit.contain,
      ),
    );
  }
}
