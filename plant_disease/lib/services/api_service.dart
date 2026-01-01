import 'dart:convert';
import 'dart:typed_data';
import 'dart:io' show File, Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

class ApiService {
  static String get baseUrl {
    if (kIsWeb) {
      return "http://127.0.0.1:8000/api";
    } else if (Platform.isAndroid) {
      return "http://10.0.2.2:8000/api";
    } else {
      return "http://127.0.0.1:8000/api";
    }
  }
  static Future<Map<String, dynamic>> uploadImage(File file) async {
    final uri = Uri.parse("$baseUrl/predict/");
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('image', file.path));

    final response = await request.send();
    final res = await http.Response.fromStream(response);

    if (res.statusCode == 200) {
      return jsonDecode(res.body);
    } else {
      throw Exception("Upload failed: ${res.statusCode} - ${res.body}");
    }
  }

  static Future<Map<String, dynamic>> uploadImageWeb(
    Uint8List fileBytes, {
    required String fileName,
  }) async {
    final uri = Uri.parse("$baseUrl/predict/");
    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      http.MultipartFile.fromBytes('image', fileBytes, filename: fileName),
    );

    final response = await request.send();
    final res = await http.Response.fromStream(response);

    if (res.statusCode == 200) {
      return jsonDecode(res.body);
    } else {
      throw Exception("Upload failed: ${res.statusCode} - ${res.body}");
    }
  }
  static Future<Map<String, dynamic>> upload(dynamic file,
      {Uint8List? bytes, String? fileName}) async {
    if (kIsWeb) {
      if (bytes == null || fileName == null) {
        throw Exception("Web upload requires bytes and fileName");
      }
      return await uploadImageWeb(bytes, fileName: fileName);
    } else {
      return await uploadImage(file);
    }
  }
}
