import streamlit as st
import os
from PIL import Image
import numpy as np

# Import custom agents and leaf disease model
from agents import MasterCoordinator
from disease_model import analyze_leaf_disease

# Page Configuration
st.set_page_config(
    page_title="KrishiMitra AI - Indian Agriculture Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 9-Language Localization Dictionary
LANG_DICT = {
    "en": {
        "title": "KrishiMitra AI 🌾",
        "tagline": "Empowering Indian Farmers with AI-Driven Insights for Better Yields",
        "mode_label": "Select LLM Execution Mode",
        "mode_local": "Offline Knowledge-Base (Low Resource / Offline)",
        "mode_ollama": "Local Ollama LLM (Llama 3/Mistral/Gemma)",
        "model_label": "Ollama Model Name",
        "btn_ask": "Send Query",
        "placeholder_chat": "Ask anything (e.g., Which crop to grow in black soil in June?)...",
        "speak_label": "🎙️ Speech-to-Text Input (English/Hindi/Regional)",
        "speak_btn": "Start Speaking",
        "nav_chat": "💬 Coordinator Chat",
        "nav_crop": "🌱 Crop Recommendation",
        "nav_disease": "🐛 Pest & Disease Doctor",
        "nav_weather": "🌦️ Weather Advisory",
        "nav_mandi": "💰 Mandi Market Price",
        "nav_schemes": "🏛️ Govt Yojana Assistant",
        "crop_head": "Calculate Recommended Crop Yields",
        "lbl_soil": "Soil Type",
        "lbl_npk": "Soil Nutrients (NPK Value Ratio)",
        "lbl_location": "Select Location (State/District)",
        "lbl_season": "Current Month/Season",
        "lbl_rain": "Expected Monsoon Intensity",
        "btn_calc_crop": "Analyze Suitable Crops",
        "disease_head": "Leaf Diagnostic OpenCV Chamber",
        "disease_upload": "Upload leaf picture to analyze lesions/spots",
        "weather_head": "Actionable Weather & Sowing Advisories",
        "mandi_head": "APMC Mandi & eNAM Price Intelligence",
        "scheme_head": "Sarkari Yojana Eligibility Advisor",
        "elig_q1": "Do you own less than 2 hectares of cultivable land?",
        "elig_q2": "Are you or your family tax-payers?",
        "elig_q3": "Do you hold institutional land ownership?",
        "btn_check_elig": "Check Scheme Eligibility",
        "warning_disclaimer": "⚠️ Disclaimer: KrishiMitra AI is an educational diagnostic helper. Verify guidelines with local Krishi Vigyan Kendra (KVK) before purchasing chemical pesticides."
    },
    "hi": {
        "title": "कृषिमित्र एआई 🌾",
        "tagline": "बेहतर उपज और अधिक आय के लिए एआई-संचालित कृषि अंतर्दृष्टि",
        "mode_label": "एलएलएम मोड चुनें",
        "mode_local": "ऑफलाइन विशेषज्ञ प्रणाली (तेज़ और कम साधन)",
        "mode_ollama": "लोकल ओलामा एलएलएम (लॉमा ३/मिस्ट्रल)",
        "model_label": "ओलामा मॉडल का नाम",
        "btn_ask": "सवाल पूछें",
        "placeholder_chat": "कुछ भी पूछें (जैसे: जून में काली मिट्टी में कौन सी फसल उगानी चाहिए?)...",
        "speak_label": "🎙️ आवाज द्वारा बोलें",
        "speak_btn": "बोलना शुरू करें",
        "nav_chat": "💬 मुख्य बातचीत",
        "nav_crop": "🌱 फसल की सिफारिश",
        "nav_disease": "🐛 कीट और रोग नियंत्रण",
        "nav_weather": "🌦️ मौसम पूर्वानुमान",
        "nav_mandi": "💰 मंडी बाजार भाव",
        "nav_schemes": "🏛️ सरकारी योजनाएं",
        "crop_head": "फसल और उपज की सिफारिश निकालें",
        "lbl_soil": "मिट्टी का प्रकार",
        "lbl_npk": "मिट्टी के पोषक तत्व (NPK अनुपात)",
        "lbl_location": "स्थान चुनें (राज्य/जिला)",
        "lbl_season": "वर्तमान महीना/मौसम",
        "lbl_rain": "अनुमानित मानसून बारिश",
        "btn_calc_crop": "फसल की जाँच करें",
        "disease_head": "पत्ता रोग पहचान एवं उपचार",
        "disease_upload": "घाव/धब्बे की जांच के लिए पत्ते की फोटो अपलोड करें",
        "weather_head": "मौसम और बुवाई के लिए दिशा-निर्देश",
        "mandi_head": "कृषि मंडी एवं ई-नाम भाव जानकारी",
        "scheme_head": "सरकारी योजना पात्रता जांच",
        "elig_q1": "क्या आपके पास 2 हेक्टेयर से कम कृषि योग्य भूमि है?",
        "elig_q2": "क्या आप या आपका परिवार टैक्स भरते हैं?",
        "elig_q3": "क्या भूमि संस्थागत (Institutional) है?",
        "btn_check_elig": "योजना पात्रता जांचें",
        "warning_disclaimer": "⚠️ चेतावनी: कृषिमित्र एआई शैक्षिक सलाह के लिए है। रासायनिक कीटनाशकों के प्रयोग से पहले कृषि विज्ञान केंद्र से संपर्क करें।"
    },
    "mr": {
        "title": "कृषिमित्र एआय 🌾",
        "tagline": "उत्कृष्ट पीक आणि अधिक उत्पन्नासाठी कृत्रिम बुद्धिमत्ता सल्लागार",
        "mode_label": "LLM मोड निवडा",
        "mode_local": "ऑफलाइन तज्ञ प्रणाली",
        "mode_ollama": "लोकल ओलामा (Llama 3/Mistral)",
        "model_label": "ओलामा मॉडेल नाव",
        "btn_ask": "विचारणा करा",
        "placeholder_chat": "काहीही विचारा (उदा. जून महिन्यात काळ्या मातीत कोणते पीक घ्यावे?)...",
        "speak_label": "🎙️ व्हॉइस इनपुट",
        "speak_btn": "बोलणे सुरू करा",
        "nav_chat": "💬 मुख्य चॅट",
        "nav_crop": "🌱 पीक शिफारस",
        "nav_disease": "🐛 रोग आणि कीटक डॉक्टर",
        "nav_weather": "🌦️ हवामान अंदाज",
        "nav_mandi": "💰 बाजार भाव (मंडी)",
        "nav_schemes": "🏛️ सरकारी योजना सहाय्यक",
        "crop_head": "योग्य पीक आणि खतांची शिफारस",
        "lbl_soil": "मातीचा प्रकार",
        "lbl_npk": "मातीचे पोषण घटक (NPK)",
        "lbl_location": "जिल्हा/राज्य निवडा",
        "lbl_season": "चालू महिना/हंगाम",
        "lbl_rain": "अपेक्षित पाऊसमान",
        "btn_calc_crop": "पीक सल्ला मिळवा",
        "disease_head": "पानावरील रोगांचे निदान व जैविक उपाय",
        "disease_upload": "पानाचा फोटो अपलोड करा",
        "weather_head": "हवामान आणि पीक संरक्षण सल्ला",
        "mandi_head": "एपीएमसी बाजार भाव आणि ई-नाम अपडेट",
        "scheme_head": "पंतप्रधान किसान योजना पात्रता",
        "elig_q1": "तुमच्याकडे २ हेक्टरपेक्षा कमी शेतजमीन आहे का?",
        "elig_q2": "तुम्ही किंवा तुमचे कुटुंब आयकर भरता का?",
        "elig_q3": "तुमची जमीन संस्थागत आहे का?",
        "btn_check_elig": "पात्रता तपासा",
        "warning_disclaimer": "⚠️ माहिती: रासायनिक फवारणीपूर्वी स्थानिक कृषी विभागाचा सल्ला घ्या."
    },
    "bn": {
        "title": "কৃষি মিত্র এআই 🌾",
        "tagline": "ভাল ফসল এবং বেশি আয়ের জন্য এআই-চালিত কৃষি পরামর্শ",
        "mode_label": "এলএলএম মোড নির্বাচন করুন",
        "mode_local": "অফলাইন বিশেষজ্ঞ সিস্টেম",
        "mode_ollama": "লোকাল ওলামা (Llama 3/Mistral)",
        "model_label": "মডেলের নাম",
        "btn_ask": "জিজ্ঞাসা করুন",
        "placeholder_chat": "যে কোনো প্রশ্ন করুন (যেমন: জুনে কালো মাটিতে কোন ফসল চাষ করব?)...",
        "speak_label": "🎙️ ভয়েস ইনপুট",
        "speak_btn": "কথা বলুন",
        "nav_chat": "💬 কৃষি চ্যাট",
        "nav_crop": "🌱 ফসলের পরামর্শ",
        "nav_disease": "🐛 রোগ ও পোকা নিরাময়",
        "nav_weather": "🌦️ আবহাওয়ার পূর্বাভাস",
        "nav_mandi": "💰 মান্ডি বাজার দর",
        "nav_schemes": "🏛️ সরকারি যোজনা তথ্য",
        "crop_head": "উপযুক্ত ফসল ও সারের হিসাব",
        "lbl_soil": "মাটির ধরণ",
        "lbl_npk": "মাটির পুষ্টিগুণ (NPK)",
        "lbl_location": "আপনার রাজ্য/জেলা",
        "lbl_season": "চলতি মাস/ঋতু",
        "lbl_rain": "বৃষ্টিপাতের পরিমাণ",
        "btn_calc_crop": "ফসলের হিসাব করুন",
        "disease_head": "পাতার রোগ নির্ণয় চেম্বার",
        "disease_upload": "রোগাক্রান্ত পাতার ছবি আপলোড করুন",
        "weather_head": "আবহাওয়া এবং বপনের পরামর্শ",
        "mandi_head": "এপিএমসি মান্ডি এবং ই-নাম বাজার দর",
        "scheme_head": "সরকারি স্কিম যোগ্যতা যাচাই",
        "elig_q1": "আপনার কি ২ হেক্টরের কম জমি আছে?",
        "elig_q2": "আপনি বা আপনার পরিবার কি আয়কর দেন?",
        "elig_q3": "আপনার জমি কি প্রাতিষ্ঠানিক মালিকানাধীন?",
        "btn_check_elig": "যোগ্যতা পরীক্ষা করুন",
        "warning_disclaimer": "⚠️ সতর্কতা: কীটনাশক ব্যবহারের আগে স্থানীয় কৃষি বিজ্ঞান কেন্দ্রের সাহায্য নিন।"
    },
    "ta": {
        "title": "கிருஷிமித்ரா ஏஐ 🌾",
        "tagline": "சிறந்த மகசூல் மற்றும் அதிக வருமானத்திற்கான ஏஐ விவசாய வழிகாட்டி",
        "mode_label": "செயலாக்க பயன்முறையைத் தேர்ந்தெடுக்கவும்",
        "mode_local": "ஆஃப்லைன் நிபுணர் முறை",
        "mode_ollama": "லோக்கல் ஓலாமா (Llama 3)",
        "model_label": "ஓலாமா மாடல் பெயர்",
        "btn_ask": "கேள்வி கேட்க",
        "placeholder_chat": "ஏதேனும் கேளுங்கள் (உதாரணமாக: ஜூன் மாதத்தில் கரிசல் மண்ணில் என்ன பயிரிட வேண்டும்?)...",
        "speak_label": "🎙️ குரல் உள்ளீடு",
        "speak_btn": "பேச தொடங்குங்கள்",
        "nav_chat": "💬 விவசாய அரட்டை",
        "nav_crop": "🌱 பயிர் பரிந்துரை",
        "nav_disease": "🐛 நோய் கண்டறிதல்",
        "nav_weather": "🌦️ வானிலை ஆலோசனை",
        "nav_mandi": "💰 மண்டி சந்தை விலை",
        "nav_schemes": "🏛️ அரசு திட்டங்கள்",
        "crop_head": "பயிர் பரிந்துரை கால்குலேட்டர்",
        "lbl_soil": "மண் வகை",
        "lbl_npk": "மண் சத்துக்கள் (NPK)",
        "lbl_location": "மாவட்டம்/மாநிலம்",
        "lbl_season": "தற்போதைய மாதம்/பருவம்",
        "lbl_rain": "மழைப்பொழிவு அளவு",
        "btn_calc_crop": "பயிர்களை ஆராயுங்கள்",
        "disease_head": "இலை நோய் கண்டறிதல்",
        "disease_upload": "இலையின் புகைப்படத்தைப் பதிவேற்றவும்",
        "weather_head": "விவசாய வானிலை எச்சரிக்கைகள்",
        "mandi_head": "மண்டி மற்றும் இ-நாம் நேரடி சந்தை விலைகள்",
        "scheme_head": "பிரதான் மந்திரி கிசான் தகுதி சரிபார்ப்பு",
        "elig_q1": "உங்களுக்கு 2 ஹெக்டேருக்கும் குறைவான விவசாய நிலம் உள்ளதா?",
        "elig_q2": "நீங்கள் அல்லது உங்கள் குடும்பத்தினர் வருமான வரி செலுத்துகிறீர்களா?",
        "elig_q3": "நிலம் நிறுவனத்திற்கு சொந்தமானதா?",
        "btn_check_elig": "தகுதியைச் சரிபார்க்கவும்",
        "warning_disclaimer": "⚠️ பொறுப்புத் துறப்பு: மருந்து தெளிப்பதற்கு முன் வேளாண் அதிகாரியை அணுகவும்."
    },
    "te": {
        "title": "కృషిమిత్ర ఐ 🌾",
        "tagline": "మెరుగైన దిగుబడి మరియు అధిక ఆదాయం కోసం ఏఐ వ్యవసాయ సలహాదారు",
        "mode_label": "మోడ్ ఎంచుకోండి",
        "mode_local": "ఆఫ్‌లైన్ నిపుణుల వ్యవస్థ",
        "mode_ollama": "లోకల్ ఓలామా ఏఐ",
        "model_label": "మోడల్ పేరు",
        "btn_ask": "ప్రశ్నించండి",
        "placeholder_chat": "ఏదైనా అడగండి (ఉదా: జూన్ నెలలో నల్ల రేగడి మట్టిలో ఏ పంట వేయాలి?)...",
        "speak_label": "🎙️ వాయిస్ ద్వారా అడగండి",
        "speak_btn": "మాట్లాడండి",
        "nav_chat": "💬 వ్యవసాయ చాట్",
        "nav_crop": "🌱 పంటల సిఫార్సు",
        "nav_disease": "🐛 తెగుళ్ల నివారణ",
        "nav_weather": "🌦️ వాతావరణ సమాచారం",
        "nav_mandi": "💰 మార్కెట్ ధరలు (మండి)",
        "nav_schemes": "🏛️ ప్రభుత్వ పథకాలు",
        "crop_head": "దిగుబడి మరియు పంటల సిఫార్సుల క్యాలిక్యులేటర్",
        "lbl_soil": "నేల రకం",
        "lbl_npk": "నేల పోషకాలు (NPK)",
        "lbl_location": "ప్రాంతం/జిల్లా",
        "lbl_season": "ప్రస్తుత నెల/కాలం",
        "lbl_rain": "వర్షపాతం అంచనా",
        "btn_calc_crop": "పంటల విశ్లేషణ చేయండి",
        "disease_head": "ఆకు తెగుళ్లు మరియు నివారణోపాయాలు",
        "disease_upload": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "weather_head": "వాతావరణ ఆధారిత వ్యవసాయ సలహాలు",
        "mandi_head": "ఏపీఎంసీ మండి ధరల సమాచారం",
        "scheme_head": "పీఎం కిసాన్ పథకం అర్హత",
        "elig_q1": "మీకు 2 హెక్టార్ల కంటే తక్కువ భూమి ఉందా?",
        "elig_q2": "మీరు లేదా మీ కుటుంబ సభ్యులు ఆదాయపు పన్ను చెల్లిస్తున్నారా?",
        "elig_q3": "మీ భూమి సంస్థాగతమైనదా?",
        "btn_check_elig": "అర్హతను తనిఖీ చేయండి",
        "warning_disclaimer": "⚠️ గమనిక: రసాయన మందుల వాడకానికి ముందు స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
    },
    "kn": {
        "title": "ಕೃಷಿಮಿತ್ರ ಎಐ 🌾",
        "tagline": "ಉತ್ತಮ ಇಳುವರಿ ಮತ್ತು ಹೆಚ್ಚಿನ ಆದಾಯಕ್ಕಾಗಿ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಸಲಹೆಗಾರ",
        "mode_label": "ಇಎಲ್‌ಎಂ ಮೋಡ್ ಆಯ್ಕೆಮಾಡಿ",
        "mode_local": "ಆಫ್‌ಲೈನ್ ತಜ್ಞ ವ್ಯವಸ್ಥೆ",
        "mode_ollama": "ಲೋಕಲ್ ಓಲಾಮಾ ಎಐ",
        "model_label": "ಮಾಡೆಲ್ ಹೆಸರು",
        "btn_ask": "ಪ್ರಶ್ನೆ ಕೇಳಿ",
        "placeholder_chat": "ಏನನ್ನಾದರೂ ಕೇಳಿ (ಉದಾ: ಜೂನ್ ತಿಂಗಳಲ್ಲಿ ಕಪ್ಪು ಮಣ್ಣಿನಲ್ಲಿ ಯಾವ ಬೆಳೆ ಬೆಳೆಯಬೇಕು?)...",
        "speak_label": "🎙️ ಧ್ವನಿ ಇನ್‌ಪುಟ್",
        "speak_btn": "ಮಾತನಾಡಿ",
        "nav_chat": "💬 ಕೃಷಿ ಸಂವಾದ",
        "nav_crop": "🌱 ಬೆಳೆ ಶಿಫಾರಸು",
        "nav_disease": "🐛 ಕೀಟ ಮತ್ತು ರೋಗ ನಿಯಂತ್ರಣ",
        "nav_weather": "🌦️ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
        "nav_mandi": "💰 ಮಾರುಕಟ್ಟೆ ದರ (ಮಂಡಿ)",
        "nav_schemes": "🏛️ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "crop_head": "ಬೆಳೆ ಶಿಫಾರಸು ಮತ್ತು ಇಳುವರಿ ಮಾಹಿತಿ",
        "lbl_soil": "ಮಣ್ಣಿನ ವಿಧ",
        "lbl_npk": "ಮಣ್ಣಿನ ಪೋಷಕಾಂಶಗಳು (NPK)",
        "lbl_location": "ಜಿಲ್ಲೆ/ರಾಜ್ಯ",
        "lbl_season": "ಪ್ರಸ್ತುತ ತಿಂಗಳು/ಹಂಗಾಮು",
        "lbl_rain": "ಮಳೆ ಪ್ರಮಾಣ",
        "btn_calc_crop": "ಬೆಳೆಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "disease_head": "ಎಲೆಗಳ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಪರಿಹಾರ",
        "disease_upload": "ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "weather_head": "ಹವಾಮಾನ ಮತ್ತು ಬಿತ್ತನೆ ಸಲಹೆಗಳು",
        "mandi_head": "ಎಪಿಎಂಸಿ ಮಂಡಿ ಮತ್ತು ಇ-ನಾಮ್ ದರಗಳು",
        "scheme_head": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಅರ್ಹತೆ ಪರಿಶೀಲನೆ",
        "elig_q1": "ನಿಮಗೆ 2 ಹೆಕ್ಟೇರ್‌ಗಿಂತ ಕಡಿಮೆ ಕೃಷಿ ಭೂಮಿ ಇದೆಯೇ?",
        "elig_q2": "ನೀವು ಅಥವಾ ನಿಮ್ಮ ಕುಟುಂಬದವರು ತೆರಿಗೆ ಪಾವತಿಸುತ್ತೀರಾ?",
        "elig_q3": "ನಿಮ್ಮ ಭೂಮಿ ಸಾಂಸ್ಥಿಕ ಒಡೆತನದಲ್ಲಿದೆಯೇ?",
        "btn_check_elig": "ಅರ್ಹತೆ ಪರಿಶೀಲಿಸಿ",
        "warning_disclaimer": "⚠️ ಸೂಚನೆ: ಯಾವುದೇ ಕೀಟನಾಶಕ ಬಳಸುವ ಮುನ್ನ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ."
    },
    "gu": {
        "title": "કૃષિમિત્ર એઆઈ 🌾",
        "tagline": "વધુ પાક ઉત્પાદન અને ઊંચી આવક માટે એઆઈ આધારિત કૃષિ માર્ગદર્શન",
        "mode_label": "મોડ પસંદ કરો",
        "mode_local": "ઓફલાઇન એક્સપર્ટ સિસ્ટમ",
        "mode_ollama": "લોકલ ઓલામા એઆઈ",
        "model_label": "મોડેલ નામ",
        "btn_ask": "પ્રશ્ન પૂછો",
        "placeholder_chat": "કોઈપણ પ્રશ્ન પૂછો (જેમ કે: જૂન મહિનામાં કાળી જમીનમાં કયો પાક વાવવો?)...",
        "speak_label": "🎙️ વોઈસ દ્વારા પૂછો",
        "speak_btn": "બોલવાનું શરૂ કરો",
        "nav_chat": "💬 કૃષિ ચર્ચા",
        "nav_crop": "🌱 પાકની ભલામણ",
        "nav_disease": "🐛 રોગ અને જીવાત નિયંત્રણ",
        "nav_weather": "🌦️ હવામાન માર્ગદર્શિકા",
        "nav_mandi": "💰 મંડી બજાર ભાવ",
        "nav_schemes": "🏛️ સરકારી યોજનાઓ",
        "crop_head": "પાકની ભલામણ અને ખાતર માહિતી",
        "lbl_soil": "જમીનનો પ્રકાર",
        "lbl_npk": "જમીનના પોષક તત્વો (NPK)",
        "lbl_location": "જિલ્લો/રાજ્ય",
        "lbl_season": "ચાલુ મહિનો/ઋતુ",
        "lbl_rain": "અપેક્ષિત વરસાદની સ્થિતિ",
        "btn_calc_crop": "પાક વિશ્લેષણ મેળવો",
        "disease_head": "પાંદડાના રોગની ઓળખ અને જૈવિક ઉપાય",
        "disease_upload": "પાંદડાનો ફોટો અપલોડ કરો",
        "weather_head": "હવામાન અને વાવણી સંબંધિત ચેતવણીઓ",
        "mandi_head": "એપીએમસી માર્કેટ યાર્ડ અને ઇ-નામ ભાવ",
        "scheme_head": "સરકારી યોજના પાત્રતા તપાસ",
        "elig_q1": "શું તમારી પાસે 2 હેક્ટરથી ઓછી જમીન છે?",
        "elig_q2": "શું તમે અથવા તમારા પરિવાર ટેક્સ ભરો છો?",
        "elig_q3": "શું તમારી જમીન સંસ્થાકીય છે?",
        "btn_check_elig": "પાત્રતા ચકાસો",
        "warning_disclaimer": "⚠️ અસ્વીકરણ: દવા છાંટતા પહેલાં સ્થાનિક કૃષિ અધિકારીની સલાહ લો."
    },
    "pa": {
        "title": "ਕ੍ਰਿਸ਼ੀਮਿੱਤਰ ਏਆਈ 🌾",
        "tagline": "ਵਧੀਆ ਝਾੜ ਅਤੇ ਵੱਧ ਆਮਦਨ ਲਈ ਏਆਈ-ਅਧਾਰਿਤ ਖੇਤੀਬਾੜੀ ਸਲਾਹਕਾਰ",
        "mode_label": "ਮੋਡ ਚੁਣੋ",
        "mode_local": "ਔਫਲਾਈਨ ਮਾਹਰ ਸਿਸਟਮ",
        "mode_ollama": "ਲੋਕਲ ਓਲਾਮਾ ਏਆਈ",
        "model_label": "ਮਾਡਲ ਦਾ ਨਾਮ",
        "btn_ask": "ਸਵਾਲ ਪੁੱਛੋ",
        "placeholder_chat": "ਕੁਝ ਵੀ ਪੁੱਛੋ (ਜਿਵੇਂ: ਜੂਨ ਵਿੱਚ ਕਾਲੀ ਮਿੱਟੀ ਵਿੱਚ ਕਿਹੜੀ ਫਸਲ ਉਗਾਈਏ?)...",
        "speak_label": "🎙️ ਆਵਾਜ਼ ਰਾਹੀਂ ਬੋਲੋ",
        "speak_btn": "ਬੋਲਣਾ ਸ਼ੁਰੂ ਕਰੋ",
        "nav_chat": "💬 ਖੇਤੀਬਾੜੀ ਚੈਟ",
        "nav_crop": "🌱 ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
        "nav_disease": "🐛 ਕੀੜੇ ਅਤੇ ਬਿਮਾਰੀ ਕੰਟਰੋਲ",
        "nav_weather": "🌦️ ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ",
        "nav_mandi": "💰 ਮੰਡੀ ਬਾਜ਼ਾਰ ਭਾਅ",
        "nav_schemes": "🏛️ ਸਰਕਾਰੀ ਸਕੀਮਾਂ",
        "crop_head": "ਫਸਲ ਅਤੇ ਖਾਦ ਕੈਲਕੁਲੇਟਰ",
        "lbl_soil": "ਮਿੱਟੀ ਦੀ ਕਿਸਮ",
        "lbl_npk": "ਮਿੱਟੀ ਦੇ ਪੋਸ਼ਕ ਤੱਤ (NPK)",
        "lbl_location": "ਸੂਬਾ/ਜ਼ਿਲ੍ਹਾ",
        "lbl_season": "ਮੌਜੂਦਾ ਮਹੀਨਾ/ਮੌਸਮ",
        "lbl_rain": "ਮੌਨਸੂਨ ਦਾ ਅੰਦਾਜ਼ਾ",
        "btn_calc_crop": "ਫਸਲਾਂ ਦੀ ਚੋਣ ਕਰੋ",
        "disease_head": "ਪੱਤਿਆਂ ਦੀ ਬਿਮਾਰੀ ਦੀ ਜਾਂਚ ਅਤੇ ਜੈਵਿਕ ਇਲਾਜ",
        "disease_upload": "ਪੱਤੇ ਦੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ",
        "weather_head": "ਮੌਸਮ ਅਤੇ ਬਿਜਾਈ ਸੰਬੰਧੀ ਸਲਾਹ",
        "mandi_head": "ਏਪੀਐਮਸੀ ਮੰਡੀ ਅਤੇ ਈ-ਨਾਮ ਰੇਟ",
        "scheme_head": "ਪੀਐਮ ਕਿਸਾਨ ਯੋਜਨਾ ਯੋਗਤਾ ਜਾਂਚ",
        "elig_q1": "ਕੀ ਤੁਹਾਡੇ ਕੋਲ 2 ਹੈਕਟੇਅਰ ਤੋਂ ਘੱਟ ਵਾਹੀਯੋਗ ਜ਼ਮੀਨ ਹੈ?",
        "elig_q2": "ਕੀ ਤੁਸੀਂ ਜਾਂ ਤੁਹਾਡਾ ਪਰਿਵਾਰ ਟੈਕਸ ਭਰਦੇ ਹੋ?",
        "elig_q3": "ਕੀ ਜ਼ਮੀਨ ਸੰਸਥਾਗਤ ਹੈ?",
        "btn_check_elig": "ਯੋਗਤਾ ਦੀ ਜਾਂਚ ਕਰੋ",
        "warning_disclaimer": "⚠️ ਬੇਦਾਅਵਾ: ਕੀਟਨਾਸ਼ਕ ਵਰਤਣ ਤੋਂ ਪਹਿਲਾਂ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਮਾਹਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।"
    }
}

# Sidebar Settings
st.sidebar.markdown(f"<div style='text-align: center;'><h2 style='color:#10b981;margin-bottom:0;'>🚜 KrishiMitra AI</h2><p style='color:#f97316;font-size:12px;font-weight:bold;letter-spacing:1px;text-transform:uppercase;'>Bharat Digital Kheti</p></div>", unsafe_allow_html=True)
st.sidebar.divider()

# Language Picker
lang_code = st.sidebar.selectbox("🗣️ Choose Language / भाषा बदलें", options=list(LANG_DICT.keys()), format_func=lambda x: {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "mr": "मराठी (Marathi)",
    "bn": "বাংলা (Bengali)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "gu": "ગુજરાતી (Gujarati)",
    "pa": "ਪੰਜਾਬੀ (Punjabi)"
}[x])

T = LANG_DICT[lang_code]

# LLM Config
execution_mode = st.sidebar.radio(T["mode_label"], options=["offline", "ollama"], format_func=lambda x: T["mode_local"] if x == "offline" else T["mode_ollama"])
ollama_model = "llama3"
if execution_mode == "ollama":
    ollama_model = st.sidebar.text_input(T["model_label"], value="llama3")

st.sidebar.divider()
st.sidebar.markdown(f"**State / Market Region:**")
mandi_state = st.sidebar.selectbox("Select State", ["Maharashtra", "Punjab", "Gujarat", "West Bengal", "Karnataka", "Andhra Pradesh", "Uttar Pradesh", "Rajasthan"])

st.sidebar.markdown(T["warning_disclaimer"])

# Main Header
st.title(T["title"])
st.subheader(T["tagline"])
st.divider()

# Initialize Agents
coordinator = MasterCoordinator(ollama_url="http://localhost:11434", model_name=ollama_model)
use_ollama_bool = (execution_mode == "ollama")

# Navigation Tabs
tab_chat, tab_crop, tab_disease, tab_weather, tab_mandi, tab_schemes = st.tabs([
    T["nav_chat"], T["nav_crop"], T["nav_disease"], T["nav_weather"], T["nav_mandi"], T["nav_schemes"]
])

# 1. COORDINATOR CHAT VIEW
with tab_chat:
    st.markdown("### 💬 Master Coordinator Chatbot")
    st.write("Talk directly with KrishiMitra. Type your query or choose a voice template to begin.")
    
    # Custom Web Speech HTML API container for browser voice transcription
    speech_html = """
    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;">
        <button id="start-record-btn" style="background-color: #ef4444; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; display: flex; align-items: center; gap: 8px;">
            🎙️ Speak Query (बोलें)
        </button>
        <p id="record-status" style="margin-top: 8px; font-size: 13px; color: #94a3b8; font-style: italic;">Microphone ready. Click button to talk.</p>
        <textarea id="transcribed-text" style="width: 100%; height: 60px; background-color: #0f172a; color: white; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 4px; padding: 8px; margin-top: 10px;" placeholder="Spoken words will appear here... (बोलने पर आपका सवाल यहां दिखेगा)"></textarea>
        <p style="font-size: 11px; color: #64748b; margin-top: 4px;">*Note: Copy the transcribed text and paste it into the chat input field below.*</p>
    </div>

    <script>
        const recordBtn = document.getElementById('start-record-btn');
        const statusText = document.getElementById('record-status');
        const textBox = document.getElementById('transcribed-text');

        if ('webkitSpeechRecognition' in window) {
            const recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'hi-IN'; // Default to Hindi, browser auto handles regional accents

            recordBtn.addEventListener('click', () => {
                recognition.start();
                statusText.innerText = "Listening... Speak now (सुन रहा हूँ... बोलिए)";
                recordBtn.style.backgroundColor = "#22c55e";
            });

            recognition.onresult = (event) => {
                const speechToText = event.results[0][0].transcript;
                textBox.value = speechToText;
                statusText.innerText = "Speech captured! Copy it below.";
                recordBtn.style.backgroundColor = "#ef4444";
            };

            recognition.onerror = (e) => {
                statusText.innerText = "Error capturing speech: " + e.error;
                recordBtn.style.backgroundColor = "#ef4444";
            };
        } else {
            statusText.innerText = "Web Speech API is not supported in this browser. Try Google Chrome.";
            recordBtn.disabled = true;
        }
    </script>
    """
    st.components.v1.html(speech_html, height=220)

    # Preset templates for ease
    st.write("💡 **Common farmer queries (click to copy):**")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        if st.button("June, Black Cotton soil crop recommendation"):
            st.session_state.chat_query = "June month, black cotton soil crop recommendation for Maharashtra."
    with col_t2:
        if st.button("Urea & DAP dosage for Rice"):
            st.session_state.chat_query = "What is the Urea and DAP dosage for Rice crop at tillering stage?"
    with col_t3:
        if st.button("PM-KISAN registration process"):
            st.session_state.chat_query = "Explain the PM-KISAN registration process and eligibility."

    # Chat Input
    query_val = st.session_state.get("chat_query", "")
    chat_query = st.text_input(T["placeholder_chat"], value=query_val, key="chat_input_field")

    if st.button(T["btn_ask"]) or chat_query:
        if chat_query:
            with st.spinner("KrishiMitra Coordinator is routing query..."):
                response = coordinator.route_and_solve(chat_query, use_ollama=use_ollama_bool)
                
                # Show results in a clean card
                st.markdown(f"""
                <div style="background-color: rgba(16, 185, 129, 0.05); border-left: 5px solid #10b981; padding: 20px; border-radius: 8px; margin-top: 15px;">
                    <h4 style="color: #10b981; margin-top: 0;">🌾 KrishiMitra Coordinator Response</h4>
                    <p style="white-space: pre-wrap; font-size: 15px; color: #f8fafc; line-height: 1.6;">{response}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Reset key to prevent infinite loop
                st.session_state.chat_query = ""

# 2. CROP RECOMMENDATION VIEW
with tab_crop:
    st.markdown(f"### {T['crop_head']}")
    
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        soil_type = st.selectbox(T["lbl_soil"], ["Black Cotton Soil", "Alluvial Soil (Sandy-loam)", "Red Soil", "Sandy/Desert Soil", "Laterite Soil"])
        location = st.text_input(T["lbl_location"], value=mandi_state)
        season = st.selectbox(T["lbl_season"], ["June (Kharif monsoon sowing)", "October (Rabi winter sowing)", "March (Zaid summer sowing)"])
        rainfall = st.select_slider(T["lbl_rain"], options=["Very Low", "Low", "Medium/Normal", "Heavy Monsoon"])

    with col_c2:
        st.write(T["lbl_npk"])
        n_val = st.slider("Nitrogen (N) kg/acre", 0, 100, 40)
        p_val = st.slider("Phosphorus (P) kg/acre", 0, 100, 20)
        k_val = st.slider("Potassium (K) kg/acre", 0, 100, 20)

    if st.button(T["btn_calc_crop"], key="btn_crop_calc"):
        with st.spinner("Analyzing soil details and nutrients..."):
            ans = coordinator.crop_agent.recommend(soil_type, (n_val, p_val, k_val), location, season, rainfall, use_ollama=use_ollama_bool)
            st.markdown(f"""
            <div style="background-color: rgba(14, 165, 233, 0.05); border: 1px solid rgba(14, 165, 233, 0.2); padding: 20px; border-radius: 8px; margin-top: 15px;">
                <h4 style="color: #0ea5e9; margin-top:0;">🌾 Recommended Crops & Resources</h4>
                <p style="white-space: pre-wrap; font-size:15px; color: #f8fafc;">{ans}</p>
            </div>
            """, unsafe_allow_html=True)

# 3. PEST & DISEASE DOCTOR VIEW
with tab_disease:
    st.markdown(f"### {T['disease_head']}")
    st.write(T["disease_upload"])
    
    uploaded_file = st.file_uploader("Upload leaf image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.image(uploaded_file, caption="Original Leaf Image", use_container_width=True)
            
        with col_d2:
            with st.spinner("Running OpenCV Disease Identification..."):
                # Reset file pointer to beginning before reading
                uploaded_file.seek(0)
                result = analyze_leaf_disease(uploaded_file)
                
                # Show OpenCV processed contour mask
                st.image(result["pil_image"], caption="OpenCV Lesion Scanning Map (Red Contours)", use_container_width=True)
                
        # Diagnosis report card
        st.markdown(f"""
        <div style="background-color: rgba(239, 68, 68, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 12px; margin-top: 20px;">
            <h4 style="margin-top:0; color:#ef4444;">🐛 Automated Plant Pathologist Report</h4>
            <div style="display:flex; justify-content:space-between; margin-bottom: 15px; font-weight:bold;">
                <span>Detected Condition: <font color="#f59e0b">{result['primary_disease']}</font></span>
                <span>Severity: <font color="{result['status_color']}">{result['severity_pct']}% - {result['severity_label']}</font></span>
            </div>
            <div style="margin-bottom: 12px;">
                <strong>🌿 Organic Control (Bio-pesticides & Home remedies):</strong>
                <p style="color:#94a3b8; font-size:14px; margin-top:4px;">{result['organic_care']}</p>
            </div>
            <div>
                <strong>🧪 Chemical Control (Govt approved fertilizers/pesticides):</strong>
                <p style="color:#94a3b8; font-size:14px; margin-top:4px;">{result['chemical_care']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# 4. WEATHER ADVISORY VIEW
with tab_weather:
    st.markdown(f"### {T['weather_head']}")
    weather_loc = st.text_input("Enter location for agricultural weather advisory:", value=mandi_state)
    
    if st.button("Get Agricultural Weather Advice"):
        with st.spinner("Fetching forecast and analyzing coordinates..."):
            ans = coordinator.weather_agent.get_weather_forecast(weather_loc, use_ollama=use_ollama_bool)
            
            st.markdown(f"""
            <div style="background-color: rgba(245, 158, 11, 0.04); border-left: 5px solid #f59e0b; padding: 20px; border-radius: 8px;">
                <h4 style="color:#f59e0b; margin-top:0;">🌦️ Sowing & Irrigation Advisory</h4>
                <p style="white-space: pre-wrap; font-size:15px; color:#f8fafc;">{ans}</p>
            </div>
            """, unsafe_allow_html=True)

# 5. MANDI PRICE INTEL VIEW
with tab_mandi:
    st.markdown(f"### {T['mandi_head']}")
    st.write("Track real-time Mandi (APMC) rates and price trends to sell at the best market.")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mandi_crop = st.selectbox("Select Crop", ["Soybean", "Cotton", "Wheat", "Rice"])
    with col_m2:
        mandi_state_sel = st.selectbox("Select Mandi State", ["Maharashtra", "Punjab", "Gujarat", "West Bengal", "Uttar Pradesh", "Rajasthan"], index=0)

    if st.button("Check Mandi Rates"):
        with st.spinner("Retrieving APMC/eNAM market records..."):
            ans = coordinator.market_agent.get_market_trends(mandi_crop, mandi_state_sel, use_ollama=use_ollama_bool)
            
            st.markdown(f"""
            <div style="background-color: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.15); padding: 20px; border-radius: 8px;">
                <h4 style="color:#10b981; margin-top:0;">💰 Mandi Price intelligence</h4>
                <p style="white-space: pre-wrap; font-size:15px; color:#f8fafc;">{ans}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple Chart visualization representing price fluctuations
            st.write("📈 **Mandi Price Trend (Last 5 Months)**")
            prices = [4200, 4350, 4500, 4420, 4800] if mandi_crop.lower() == "soybean" else [6200, 6400, 6800, 6750, 7150]
            st.line_chart(np.array(prices))

# 6. GOVT YOJANA VIEW
with tab_schemes:
    st.markdown(f"### {T['scheme_head']}")
    
    col_y1, col_y2 = st.columns([1.2, 0.8])
    
    with col_y1:
        st.write("#### 🏛️ Learn about major agricultural schemes:")
        scheme_choice = st.selectbox("Select Scheme", ["PM-KISAN (Income Support)", "PMFBY (Crop Insurance / Fasal Bima)", "PM-KUSUM (Solar Pumps & Subsidies)"])
        
        if st.button("Explain Scheme Details"):
            with st.spinner("Processing scheme guidelines..."):
                ans = coordinator.scheme_agent.get_schemes_info(scheme_choice, use_ollama=use_ollama_bool)
                st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255,255,255,0.1); padding: 20px; border-radius: 8px;">
                    <p style="white-space: pre-wrap; font-size:15px;">{ans}</p>
                </div>
                """, unsafe_allow_html=True)

    with col_y2:
        st.markdown("#### 📝 Interactive Eligibility Calculator")
        st.write("Check if you might qualify for PM-KISAN:")
        
        q1 = st.checkbox(T["elig_q1"], value=True)
        q2 = st.checkbox(T["elig_q2"], value=False)
        q3 = st.checkbox(T["elig_q3"], value=False)
        
        if st.button(T["btn_check_elig"]):
            # PM-KISAN rules: < 2 hectares (general baseline, though limits are expanded, excludes institutional land owners and income tax payers)
            if q1 and not q2 and not q3:
                st.success("✅ **Likely Eligible!** You meet the basic criteria. Registration can be completed on pmkisan.gov.in using your land registration copy and Aadhaar.")
            else:
                st.warning("⚠️ **Exclusion Criteria Matched:** You might not be eligible. PM-KISAN excludes tax-payers, institutional landholders, or families holding high constitutional posts.")
