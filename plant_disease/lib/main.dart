import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/predict_screen.dart';
import 'screens/graph_screen.dart';

void main() {
  runApp(const PlantDiseaseApp());
}

class PlantDiseaseApp extends StatelessWidget {
  const PlantDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Plant Disease Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFF28a745),
        fontFamily: 'Poppins',
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: Colors.black87),
        ),
      ),
      // 👇 ensure app *always starts* at HomeScreen
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        // you can add future pages here
        '/predict': (context) => const PredictScreen(),
        '/graph': (context) => const GraphScreen(),
      },
    );
  }
}
