import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  final picker = ImagePicker();

  Future getImage() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
      } else {
        print('No image selected.');
      }
    });
  }

  Future<void> uploadImage(File image) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://your-server-url/convert'),
    );
    request.files.add(await http.MultipartFile.fromPath('image', image.path));
    final response = await request.send();

    if (response.statusCode == 200) {
      final responseData = await response.stream.bytesToString();
      final result = jsonDecode(responseData);
      // Handle the result
    } else {
      print('Failed to upload image.');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AI Toon Photo Creator'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null
                ? Text('No image selected.')
                : Image.file(_image!),
            ElevatedButton(
              onPressed: getImage,
              child: Text('Pick Image'),
            ),
            ElevatedButton(
              onPressed: () {
                if (_image != null) {
                  uploadImage(_image!);
                }
              },
              child: Text('Convert to Toon'),
            ),
          ],
        ),
      ),
    );
  }
}
