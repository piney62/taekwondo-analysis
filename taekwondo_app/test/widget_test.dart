import 'package:flutter_test/flutter_test.dart';
import 'package:taekwondo_app/main.dart';

void main() {
  testWidgets('App renders without crashing', (WidgetTester tester) async {
    await tester.pumpWidget(const TaekwondoApp());
    expect(find.text('ITF Taekwondo Analysis'), findsOneWidget);
  });
}
