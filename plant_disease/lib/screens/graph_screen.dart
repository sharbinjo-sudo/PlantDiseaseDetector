import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../widgets/navbar.dart';

class GraphScreen extends StatelessWidget {
  const GraphScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF6FAF7),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const NavBar(activePage: "Graph"),
            const SizedBox(height: 30),
            const Text(
              "Model Performance Graphs",
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Color(0xFF0C4128),
              ),
            ),
            const SizedBox(height: 40),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  _buildGraphCard(
                    title: "Training vs Validation Accuracy",
                    yLabel: "Accuracy (%)",
                    color1: const Color(0xFF28A745),
                    color2: Colors.lightBlue,
                    data1: [70, 78, 84, 88, 91, 93, 94, 95, 96, 97],
                    data2: [65, 74, 80, 85, 88, 89, 90, 91, 91.5, 92],
                  ),
                  const SizedBox(height: 40),

                  _buildGraphCard(
                    title: "Training vs Validation Loss",
                    yLabel: "Loss",
                    color1: Colors.redAccent,
                    color2: Colors.orangeAccent,
                    data1: [1.0, 0.82, 0.65, 0.52, 0.42, 0.36, 0.31, 0.28, 0.26, 0.24],
                    data2: [1.1, 0.9, 0.75, 0.62, 0.5, 0.45, 0.42, 0.40, 0.39, 0.38],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 60),
          ],
        ),
      ),
    );
  }

Widget _buildGraphCard({
  required String title,
  required List<double> data1,
  required List<double> data2,
  required Color color1,
  required Color color2,
  required String yLabel,
}) {
  return Card(
    elevation: 5,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    color: Colors.white,
    child: Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 300,
            child: LineChart(
              LineChartData(
                gridData: FlGridData(
                  show: true,
                  horizontalInterval:
                      yLabel.contains("Accuracy") ? 5 : 0.1,
                  getDrawingHorizontalLine: (value) => FlLine(
                    color: Colors.grey.shade300,
                    strokeWidth: 1,
                  ),
                ),
                titlesData: FlTitlesData(
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      interval: 1,
                      reservedSize: 32,
                      getTitlesWidget: (value, meta) {
                        final index = value.toInt();
                        if (index < 0 || index >= data1.length) {
                          return const SizedBox.shrink();
                        }
                        return Text(
                          "E${index + 1}",
                          style: const TextStyle(fontSize: 12),
                        );
                      },
                    ),
                  ),
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 40,
                      getTitlesWidget: (value, meta) => Text(
                        yLabel.contains("Accuracy")
                            ? "${value.toInt()}%"
                            : value.toStringAsFixed(1),
                        style: const TextStyle(fontSize: 11),
                      ),
                    ),
                  ),
                  topTitles:
                      AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles:
                      AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
                borderData: FlBorderData(
                  show: true,
                  border: Border.all(color: Colors.black12),
                ),
                minX: 0,
                maxX: (data1.length - 1).toDouble(),
                minY: yLabel.contains("Accuracy") ? 60 : 0.2,
                maxY: yLabel.contains("Accuracy") ? 100 : 1.2,
                lineBarsData: [
                  LineChartBarData(
                    spots: List.generate(
                      data1.length,
                      (i) => FlSpot(i.toDouble(), data1[i]),
                    ),
                    isCurved: true,
                    color: color1,
                    barWidth: 3,
                    belowBarData: BarAreaData(
                      show: true,
                      color: color1.withOpacity(0.2),
                    ),
                    dotData: FlDotData(show: false),
                  ),
                  LineChartBarData(
                    spots: List.generate(
                      data2.length,
                      (i) => FlSpot(i.toDouble(), data2[i]),
                    ),
                    isCurved: true,
                    color: color2,
                    barWidth: 3,
                    belowBarData: BarAreaData(
                      show: true,
                      color: color2.withOpacity(0.15),
                    ),
                    dotData: FlDotData(show: false),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _legendDot(color1, "Training"),
              const SizedBox(width: 20),
              _legendDot(color2, "Validation"),
            ],
          ),
        ],
      ),
    ),
  );
}


  Widget _legendDot(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 14,
          height: 14,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(color: Colors.black87)),
      ],
    );
  }
}
