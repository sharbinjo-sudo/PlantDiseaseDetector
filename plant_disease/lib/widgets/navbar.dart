import 'package:flutter/material.dart';

class NavBar extends StatelessWidget {
  final String activePage;
  const NavBar({super.key, required this.activePage});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
      color: Colors.white,
      child: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth < 600) {
            return Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Plant',
                  style: TextStyle(
                    color: Color(0xFF28a745),
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                PopupMenuButton<String>(
                  icon: const Icon(Icons.menu, color: Colors.black87),
                  onSelected: (route) {
                    if (ModalRoute.of(context)?.settings.name != route) {
                      Navigator.pushNamed(context, route);
                    }
                  },
                  itemBuilder: (context) => [
                    PopupMenuItem(
                      value: '/',
                      child: Text(
                        "Home",
                        style: TextStyle(
                          color: activePage == "Home"
                              ? const Color(0xFF28a745)
                              : Colors.black87,
                        ),
                      ),
                    ),
                    PopupMenuItem(
                      value: '/predict',
                      child: Text(
                        "Predict",
                        style: TextStyle(
                          color: activePage == "Predict"
                              ? const Color(0xFF28a745)
                              : Colors.black87,
                        ),
                      ),
                    ),
                    PopupMenuItem(
                      value: '/graph',
                      child: Text(
                        "Graph",
                        style: TextStyle(
                          color: activePage == "Graph"
                              ? const Color(0xFF28a745)
                              : Colors.black87,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            );
          }

          return Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Plant',
                style: TextStyle(
                  color: Color(0xFF28a745),
                  fontSize: 28,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Row(
                children: [
                  _navItem(context, "Home", "/", active: activePage == "Home"),
                  _navItem(context, "Predict", "/predict",
                      active: activePage == "Predict"),
                  _navItem(context, "Graph", "/graph",
                      active: activePage == "Graph"),
                ],
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _navItem(BuildContext context, String title, String route,
      {bool active = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: InkWell(
        onTap: () {
          if (ModalRoute.of(context)?.settings.name != route) {
            Navigator.pushNamed(context, route);
          }
        },
        child: Text(
          title,
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w500,
            color: active ? const Color(0xFF28a745) : Colors.black87,
          ),
        ),
      ),
    );
  }
}
