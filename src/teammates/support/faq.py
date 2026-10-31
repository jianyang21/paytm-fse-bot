"""FAQ RAG (T21): a small FAQ set and a retriever behind an interface.

TfidfRetriever is the stdlib fallback (TF-IDF cosine over 5-letter word stems). An embedding
retriever only has to implement Retriever.search. The FAQ answer is returned verbatim, so an
FAQ reply is grounded in the FAQ text by construction; the resolver uses it only when the
retrieval is confident AND the question is generic (no account-specific action or amount).
"""

import math
import re
from collections import Counter
from typing import Any, Protocol

FAQS: list[dict[str, str]] = [
    {"id": "settle-time", "topic": "settlements",
     "q": "When will I get my settlement? How long does settlement take to reach my bank?",
     "a": "Payments you receive are settled to your registered bank account on the next "
          "working day (T+1). Settlements for payments received on a Sunday or bank holiday "
          "are credited on the next working day after that."},
    {"id": "settle-bank", "topic": "settlements",
     "q": "How do I change the bank account where my settlements are credited?",
     "a": "Open the Paytm for Business app, go to Settings > Bank Account, and add the new "
          "account. It is verified with a small test credit and used for settlements once "
          "the verification succeeds."},
    {"id": "settle-statement", "topic": "settlements",
     "q": "Where can I download my settlement statement or payment report?",
     "a": "In the Paytm for Business app, open Payments > Settlements and tap Download to get "
          "a statement for any date range."},
    {"id": "refund-how", "topic": "refunds",
     "q": "How do I give a refund to my customer? How to issue a refund?",
     "a": "In the Paytm for Business app, open the payment under Payments, tap Refund, enter "
          "the amount and confirm. Full and partial refunds are both supported."},
    {"id": "refund-time", "topic": "refunds",
     "q": "How many days does a refund take to reach the customer?",
     "a": "After you issue a refund, the customer's bank usually credits it within 5 to 7 "
          "working days."},
    {"id": "failed-debit", "topic": "failed payments",
     "q": "Customer says money was debited but the payment failed. What happens to the money?",
     "a": "If a payment fails after the customer's account is debited, the bank reverses the "
          "amount to the customer automatically, usually within 5 working days. You do not "
          "need to issue a refund for a failed payment."},
    {"id": "failed-why", "topic": "failed payments",
     "q": "Why do customer payments fail? Common reasons for failed UPI payments",
     "a": "Payments usually fail because of a bank server timeout, a wrong UPI PIN, the "
          "customer's daily limit being reached, or weak internet. Ask the customer to retry "
          "after a few minutes."},
    {"id": "kyc-docs", "topic": "KYC",
     "q": "What documents do I need for KYC? Which documents are required to complete KYC?",
     "a": "You need your PAN, an Aadhaar-linked mobile number for OTP, and a business proof "
          "such as GST registration, Udyam certificate or shop licence."},
    {"id": "kyc-pending", "topic": "KYC",
     "q": "What happens if my KYC is pending or incomplete? Why is my KYC not done?",
     "a": "Until KYC is complete, settlements may be limited. Finish KYC in the app under "
          "Profile > KYC; most verifications complete within 2 working days."},
    {"id": "soundbox-silent", "topic": "devices",
     "q": "My Soundbox is not announcing payments, it is silent. How do I fix the Soundbox?",
     "a": "Check that the Soundbox is charged and switched on, press the volume up button, "
          "and make sure it shows network signal. If it still stays silent, restart it by "
          "holding the power button for 10 seconds."},
    {"id": "soundbox-language", "topic": "devices",
     "q": "How do I change the language of my Soundbox announcements?",
     "a": "Press and hold the language button on the Soundbox to cycle through the available "
          "languages; it announces the selected language each time."},
    {"id": "device-new", "topic": "devices",
     "q": "How do I log in on a new phone or device? Changed my phone or SIM",
     "a": "Install the Paytm for Business app on the new phone and log in with your "
          "registered mobile number and OTP. For your security, some features stay locked "
          "for 24 hours after a new device login."},
]

# Every answer in each merchant language the desk replies in. Sent verbatim like the English,
# so an FAQ reply stays grounded in FAQ text; each keeps exactly the English answer's numbers
# (tested). App menu names stay in English, as the app shows them. Drafted for the POC: have a
# native speaker review them before production.
TRANSLATIONS: dict[str, dict[str, str]] = {
    "settle-time": {
        "hi": "आपको मिले भुगतान अगले कार्य दिवस (T+1) पर आपके रजिस्टर्ड बैंक खाते में सेटल किए "
              "जाते हैं। रविवार या बैंक अवकाश के दिन मिले भुगतानों का सेटलमेंट उसके बाद वाले "
              "अगले कार्य दिवस पर जमा होता है।",
        "ta": "நீங்கள் பெறும் பணம் அடுத்த வேலை நாளில் (T+1) உங்கள் பதிவு செய்யப்பட்ட வங்கிக் "
              "கணக்கில் செட்டில் செய்யப்படும். ஞாயிற்றுக்கிழமை அல்லது வங்கி விடுமுறை நாளில் "
              "பெறப்பட்ட பணத்தின் செட்டில்மென்ட் அதற்குப் பிறகு வரும் அடுத்த வேலை நாளில் வரவு "
              "வைக்கப்படும்.",
        "kn": "ನೀವು ಸ್ವೀಕರಿಸುವ ಪಾವತಿಗಳು ಮುಂದಿನ ಕೆಲಸದ ದಿನ (T+1) ನಿಮ್ಮ ನೋಂದಾಯಿತ ಬ್ಯಾಂಕ್ ಖಾತೆಗೆ "
              "ಸೆಟಲ್ ಆಗುತ್ತವೆ. ಭಾನುವಾರ ಅಥವಾ ಬ್ಯಾಂಕ್ ರಜಾದಿನದಂದು ಸ್ವೀಕರಿಸಿದ ಪಾವತಿಗಳ ಸೆಟಲ್‌ಮೆಂಟ್ "
              "ಅದರ ನಂತರದ ಮುಂದಿನ ಕೆಲಸದ ದಿನ ಜಮೆಯಾಗುತ್ತದೆ."},
    "settle-bank": {
        "hi": "Paytm for Business ऐप खोलें, Settings > Bank Account में जाएं और नया खाता जोड़ें। "
              "एक छोटे टेस्ट क्रेडिट से उसकी पुष्टि की जाती है, और पुष्टि सफल होने के बाद "
              "सेटलमेंट उसी खाते में आते हैं।",
        "ta": "Paytm for Business செயலியைத் திறந்து, Settings > Bank Account சென்று புதிய "
              "கணக்கைச் சேர்க்கவும். ஒரு சிறிய சோதனை வரவு மூலம் அது சரிபார்க்கப்படும்; "
              "சரிபார்ப்பு வெற்றியடைந்ததும் செட்டில்மென்ட்கள் அந்தக் கணக்கிற்கு வரும்.",
        "kn": "Paytm for Business ಆ್ಯಪ್ ತೆರೆಯಿರಿ, Settings > Bank Account ಗೆ ಹೋಗಿ ಹೊಸ ಖಾತೆಯನ್ನು "
              "ಸೇರಿಸಿ. ಸಣ್ಣ ಪರೀಕ್ಷಾ ಜಮೆಯ ಮೂಲಕ ಅದನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತದೆ; ಪರಿಶೀಲನೆ ಯಶಸ್ವಿಯಾದ "
              "ನಂತರ ಸೆಟಲ್‌ಮೆಂಟ್‌ಗಳು ಆ ಖಾತೆಗೆ ಬರುತ್ತವೆ."},
    "settle-statement": {
        "hi": "Paytm for Business ऐप में Payments > Settlements खोलें और किसी भी तारीख की अवधि का "
              "स्टेटमेंट पाने के लिए Download पर टैप करें।",
        "ta": "Paytm for Business செயலியில் Payments > Settlements திறந்து, எந்தத் தேதி "
              "வரம்பிற்கான அறிக்கையையும் பெற Download என்பதைத் தட்டவும்.",
        "kn": "Paytm for Business ಆ್ಯಪ್‌ನಲ್ಲಿ Payments > Settlements ತೆರೆಯಿರಿ ಮತ್ತು ಯಾವುದೇ ದಿನಾಂಕ "
              "ಅವಧಿಯ ಸ್ಟೇಟ್‌ಮೆಂಟ್ ಪಡೆಯಲು Download ಒತ್ತಿರಿ."},
    "refund-how": {
        "hi": "Paytm for Business ऐप में Payments के अंदर वह भुगतान खोलें, Refund पर टैप करें, "
              "राशि डालें और पुष्टि करें। पूरा और आंशिक, दोनों तरह का रिफंड किया जा सकता है।",
        "ta": "Paytm for Business செயலியில் Payments பகுதியில் அந்தப் பணப்பரிவர்த்தனையைத் "
              "திறந்து, Refund தட்டி, தொகையை உள்ளிட்டு உறுதிப்படுத்தவும். முழு மற்றும் பகுதி "
              "ரீஃபண்ட் இரண்டும் செய்யலாம்.",
        "kn": "Paytm for Business ಆ್ಯಪ್‌ನಲ್ಲಿ Payments ಅಡಿಯಲ್ಲಿ ಆ ಪಾವತಿಯನ್ನು ತೆರೆಯಿರಿ, Refund "
              "ಒತ್ತಿ, ಮೊತ್ತವನ್ನು ನಮೂದಿಸಿ ಮತ್ತು ದೃಢೀಕರಿಸಿ. ಪೂರ್ಣ ಮತ್ತು ಭಾಗಶಃ ಎರಡೂ ರೀತಿಯ ರಿಫಂಡ್ "
              "ಮಾಡಬಹುದು."},
    "refund-time": {
        "hi": "रिफंड करने के बाद ग्राहक का बैंक आमतौर पर 5 से 7 कार्य दिवसों में उसे जमा कर देता है।",
        "ta": "நீங்கள் ரீஃபண்ட் செய்த பிறகு, வாடிக்கையாளரின் வங்கி பொதுவாக 5 முதல் 7 வேலை "
              "நாட்களுக்குள் அதை வரவு வைக்கும்.",
        "kn": "ನೀವು ರಿಫಂಡ್ ಮಾಡಿದ ನಂತರ, ಗ್ರಾಹಕರ ಬ್ಯಾಂಕ್ ಸಾಮಾನ್ಯವಾಗಿ 5 ರಿಂದ 7 ಕೆಲಸದ ದಿನಗಳೊಳಗೆ "
              "ಅದನ್ನು ಜಮೆ ಮಾಡುತ್ತದೆ."},
    "failed-debit": {
        "hi": "अगर ग्राहक के खाते से पैसे कटने के बाद भुगतान फेल हो जाए, तो बैंक वह राशि अपने आप "
              "ग्राहक को वापस कर देता है, आमतौर पर 5 कार्य दिवसों में। फेल हुए भुगतान के लिए "
              "आपको रिफंड करने की ज़रूरत नहीं है।",
        "ta": "வாடிக்கையாளரின் கணக்கில் பணம் கழிக்கப்பட்ட பிறகு பணப்பரிவர்த்தனை "
              "தோல்வியடைந்தால், வங்கி அந்தத் தொகையைத் தானாகவே வாடிக்கையாளருக்குத் திருப்பி "
              "அனுப்பும், பொதுவாக 5 வேலை நாட்களுக்குள். தோல்வியடைந்த பணப்பரிவர்த்தனைக்கு "
              "நீங்கள் ரீஃபண்ட் செய்ய வேண்டியதில்லை.",
        "kn": "ಗ್ರಾಹಕರ ಖಾತೆಯಿಂದ ಹಣ ಕಡಿತವಾದ ನಂತರ ಪಾವತಿ ವಿಫಲವಾದರೆ, ಬ್ಯಾಂಕ್ ಆ ಮೊತ್ತವನ್ನು "
              "ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಗ್ರಾಹಕರಿಗೆ ಹಿಂತಿರುಗಿಸುತ್ತದೆ, ಸಾಮಾನ್ಯವಾಗಿ 5 ಕೆಲಸದ ದಿನಗಳೊಳಗೆ. "
              "ವಿಫಲವಾದ ಪಾವತಿಗೆ ನೀವು ರಿಫಂಡ್ ಮಾಡುವ ಅಗತ್ಯವಿಲ್ಲ."},
    "failed-why": {
        "hi": "भुगतान आमतौर पर बैंक सर्वर के टाइमआउट, गलत UPI PIN, ग्राहक की दैनिक सीमा पूरी हो "
              "जाने या कमज़ोर इंटरनेट की वजह से फेल होते हैं। ग्राहक से कुछ मिनट बाद फिर से "
              "कोशिश करने को कहें।",
        "ta": "வங்கி சர்வர் நேரம் முடிதல், தவறான UPI PIN, வாடிக்கையாளரின் தினசரி வரம்பு "
              "எட்டப்பட்டது, அல்லது பலவீனமான இணையம் ஆகிய காரணங்களால் பொதுவாகப் "
              "பணப்பரிவர்த்தனைகள் தோல்வியடைகின்றன. சில நிமிடங்களுக்குப் பிறகு மீண்டும் "
              "முயற்சிக்குமாறு வாடிக்கையாளரிடம் சொல்லுங்கள்.",
        "kn": "ಬ್ಯಾಂಕ್ ಸರ್ವರ್ ಟೈಮ್‌ಔಟ್, ತಪ್ಪಾದ UPI PIN, ಗ್ರಾಹಕರ ದೈನಂದಿನ ಮಿತಿ ಮುಗಿದಿರುವುದು ಅಥವಾ "
              "ದುರ್ಬಲ ಇಂಟರ್ನೆಟ್‌ನಿಂದ ಸಾಮಾನ್ಯವಾಗಿ ಪಾವತಿಗಳು ವಿಫಲವಾಗುತ್ತವೆ. ಕೆಲವು ನಿಮಿಷಗಳ ನಂತರ "
              "ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಲು ಗ್ರಾಹಕರಿಗೆ ಹೇಳಿ."},
    "kyc-docs": {
        "hi": "आपको अपना PAN, OTP के लिए Aadhaar से जुड़ा मोबाइल नंबर, और व्यवसाय का कोई प्रमाण "
              "जैसे GST रजिस्ट्रेशन, Udyam सर्टिफिकेट या दुकान का लाइसेंस चाहिए।",
        "ta": "உங்கள் PAN, OTP-க்காக Aadhaar உடன் இணைக்கப்பட்ட மொபைல் எண், மற்றும் GST பதிவு, "
              "Udyam சான்றிதழ் அல்லது கடை உரிமம் போன்ற வணிகச் சான்று தேவை.",
        "kn": "ನಿಮ್ಮ PAN, OTP ಗಾಗಿ Aadhaar ಗೆ ಲಿಂಕ್ ಆಗಿರುವ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ, ಮತ್ತು GST ನೋಂದಣಿ, "
              "Udyam ಪ್ರಮಾಣಪತ್ರ ಅಥವಾ ಅಂಗಡಿ ಪರವಾನಗಿಯಂತಹ ವ್ಯವಹಾರದ ಪುರಾವೆ ಬೇಕು."},
    "kyc-pending": {
        "hi": "KYC पूरा होने तक सेटलमेंट सीमित हो सकते हैं। ऐप में Profile > KYC में जाकर KYC पूरा "
              "करें; ज़्यादातर सत्यापन 2 कार्य दिवसों में पूरे हो जाते हैं।",
        "ta": "KYC முடியும் வரை செட்டில்மென்ட்கள் வரம்புக்குட்படலாம். செயலியில் Profile > KYC "
              "பகுதியில் KYC-ஐ முடிக்கவும்; பெரும்பாலான சரிபார்ப்புகள் 2 வேலை நாட்களுக்குள் "
              "முடிவடையும்.",
        "kn": "KYC ಪೂರ್ಣಗೊಳ್ಳುವವರೆಗೆ ಸೆಟಲ್‌ಮೆಂಟ್‌ಗಳು ಸೀಮಿತವಾಗಿರಬಹುದು. ಆ್ಯಪ್‌ನಲ್ಲಿ Profile > KYC "
              "ಅಡಿಯಲ್ಲಿ KYC ಪೂರ್ಣಗೊಳಿಸಿ; ಹೆಚ್ಚಿನ ಪರಿಶೀಲನೆಗಳು 2 ಕೆಲಸದ ದಿನಗಳೊಳಗೆ ಮುಗಿಯುತ್ತವೆ."},
    "soundbox-silent": {
        "hi": "देखें कि Soundbox चार्ज है और चालू है, वॉल्यूम बढ़ाने वाला बटन दबाएं, और पक्का करें कि "
              "उसमें नेटवर्क सिग्नल दिख रहा है। अगर फिर भी आवाज़ न आए, तो पावर बटन 10 सेकंड तक "
              "दबाकर उसे रीस्टार्ट करें।",
        "ta": "Soundbox சார்ஜ் செய்யப்பட்டு இயக்கத்தில் உள்ளதா என்று பார்த்து, ஒலி அதிகரிக்கும் "
              "பொத்தானை அழுத்தி, நெட்வொர்க் சிக்னல் காட்டுகிறதா என்று உறுதிசெய்யவும். அப்போதும் "
              "ஒலிக்கவில்லை என்றால், பவர் பொத்தானை 10 வினாடிகள் அழுத்திப் பிடித்து மறுதொடக்கம் "
              "செய்யவும்.",
        "kn": "Soundbox ಚಾರ್ಜ್ ಆಗಿದೆಯೇ ಮತ್ತು ಆನ್ ಆಗಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ, ವಾಲ್ಯೂಮ್ ಹೆಚ್ಚಿಸುವ ಬಟನ್ "
              "ಒತ್ತಿ, ಮತ್ತು ಅದು ನೆಟ್‌ವರ್ಕ್ ಸಿಗ್ನಲ್ ತೋರಿಸುತ್ತಿದೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ. ಆಗಲೂ ಧ್ವನಿ "
              "ಬರದಿದ್ದರೆ, ಪವರ್ ಬಟನ್ ಅನ್ನು 10 ಸೆಕೆಂಡುಗಳ ಕಾಲ ಹಿಡಿದು ಅದನ್ನು ಮರುಪ್ರಾರಂಭಿಸಿ."},
    "soundbox-language": {
        "hi": "उपलब्ध भाषाओं में से चुनने के लिए Soundbox का भाषा बटन दबाकर रखें; हर बार वह चुनी "
              "गई भाषा बोलकर बताता है।",
        "ta": "கிடைக்கும் மொழிகளுக்கு இடையே மாற Soundbox-இல் உள்ள மொழி பொத்தானை அழுத்திப் "
              "பிடிக்கவும்; ஒவ்வொரு முறையும் தேர்ந்தெடுத்த மொழியை அது அறிவிக்கும்.",
        "kn": "ಲಭ್ಯವಿರುವ ಭಾಷೆಗಳ ನಡುವೆ ಬದಲಾಯಿಸಲು Soundbox ನ ಭಾಷಾ ಬಟನ್ ಅನ್ನು ಒತ್ತಿ ಹಿಡಿಯಿರಿ; ಪ್ರತಿ "
              "ಬಾರಿಯೂ ಆಯ್ಕೆಮಾಡಿದ ಭಾಷೆಯನ್ನು ಅದು ಘೋಷಿಸುತ್ತದೆ."},
    "device-new": {
        "hi": "नए फ़ोन पर Paytm for Business ऐप इंस्टॉल करें और अपने रजिस्टर्ड मोबाइल नंबर और OTP से "
              "लॉग इन करें। आपकी सुरक्षा के लिए, नए डिवाइस पर लॉग इन के बाद कुछ सुविधाएं 24 घंटे "
              "तक बंद रहती हैं।",
        "ta": "புதிய போனில் Paytm for Business செயலியை நிறுவி, உங்கள் பதிவு செய்யப்பட்ட மொபைல் "
              "எண் மற்றும் OTP மூலம் உள்நுழையவும். உங்கள் பாதுகாப்புக்காக, புதிய சாதனத்தில் "
              "உள்நுழைந்த பிறகு சில வசதிகள் 24 மணிநேரம் பூட்டப்பட்டிருக்கும்.",
        "kn": "ಹೊಸ ಫೋನ್‌ನಲ್ಲಿ Paytm for Business ಆ್ಯಪ್ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಿ ಮತ್ತು ನಿಮ್ಮ ನೋಂದಾಯಿತ ಮೊಬೈಲ್ "
              "ಸಂಖ್ಯೆ ಹಾಗೂ OTP ಬಳಸಿ ಲಾಗಿನ್ ಮಾಡಿ. ನಿಮ್ಮ ಸುರಕ್ಷತೆಗಾಗಿ, ಹೊಸ ಸಾಧನದಲ್ಲಿ ಲಾಗಿನ್ ಆದ ನಂತರ "
              "ಕೆಲವು ಸೌಲಭ್ಯಗಳು 24 ಗಂಟೆಗಳ ಕಾಲ ಲಾಕ್ ಆಗಿರುತ್ತವೆ."},
}


def answer(faq: dict[str, str], lang: str) -> str:
    """The FAQ answer in the merchant's language (English if there is no translation)."""
    return TRANSLATIONS.get(faq["id"], {}).get(lang, faq["a"])


STOP = {"a", "an", "the", "is", "are", "am", "i", "my", "me", "we", "our", "you", "your", "to",
        "of", "for", "in", "on", "at", "and", "or", "do", "does", "did", "how", "what", "when",
        "where", "why", "which", "can", "will", "it", "this", "that", "be", "with", "from",
        "not", "has", "have", "was", "were", "get", "got", "please", "hi", "hello", "sir",
        "madam"}


def tokens(text: str) -> list[str]:
    return [w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 1]


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[tuple[dict[str, str], float]]:
        """Best FAQs first, each with a confidence in 0..1."""
        ...


class TfidfRetriever:
    def __init__(self, faqs: list[dict[str, str]] = FAQS) -> None:
        self.faqs = faqs
        docs = [Counter(tokens(f["q"] + " " + f["topic"])) for f in faqs]
        df = Counter(t for d in docs for t in d)
        self.idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter[str]) -> dict[str, float]:
        v = {t: n * self.idf.get(t, 0.0) for t, n in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[dict[str, str], float]]:
        q = self._vec(Counter(tokens(query)))
        scored = [(f, round(sum(w * v.get(t, 0.0) for t, w in q.items()), 3))
                  for f, v in zip(self.faqs, self.vecs, strict=True)]
        return sorted(scored, key=lambda p: -p[1])[:k]


def faq_match(retriever: Retriever, text: str, generic: bool,
              cfg: dict[str, Any]) -> dict[str, Any]:
    """{answerable, faq (best hit or None), confidence}. Answerable only if generic AND the
    retrieval is confident; the answer is then the FAQ text, verbatim."""
    hits = retriever.search(text, 1)
    faq, conf = hits[0] if hits else (None, 0.0)
    return {"answerable": generic and conf >= cfg["min_confidence"],
            "faq": faq if conf >= cfg["context_min"] else None, "confidence": conf}
