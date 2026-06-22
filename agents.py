import requests
import json

class BaseAgent:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3"):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def call_ollama(self, prompt, system_prompt=""):
        """Calls local Ollama instance if available, otherwise raises ConnectionError."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=8)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise ConnectionError(f"Ollama returned status code {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Ollama unavailable: {str(e)}")

# 1. Crop Recommendation Agent
class CropRecommendationAgent(BaseAgent):
    def get_fallback_recommendation(self, soil_type, npk, location, season, rainfall):
        n, p, k = npk
        soil_type = soil_type.lower()
        season = season.lower()
        
        # Expert logic rules for Indian states & seasons
        if "black" in soil_type or "cotton" in soil_type:
            if "june" in season or "kharif" in season or "monsoon" in season:
                return {
                    "crop": "Cotton (Kapas) & Soybean Intercrop",
                    "yield": "12-15 quintals per acre for Cotton; 8-10 quintals for Soybean.",
                    "resources": "NPK 40:20:20 kg/acre, 2 bags of DAP, and quality BT Cotton seeds. Cotton needs deep root space.",
                    "rotation": "Follow with Chickpea (Chana) or Wheat in Rabi (October) to restore soil nitrogen.",
                    "tips": "Sow when monsoon rainfall reaches 50-60mm. Avoid waterlogging."
                }
            else:
                return {
                    "crop": "Sorghum (Jowar) or Chickpea (Chana)",
                    "yield": "8-10 quintals per acre.",
                    "resources": "NPK 20:10:10 kg/acre, thrives on residual moisture in black cotton soil.",
                    "rotation": "Follow with Green Gram (Moong) in summer.",
                    "tips": "Ensure shallow tilling before Rabi sowing to conserve soil moisture."
                }
        elif "alluvial" in soil_type:
            if "june" in season or "kharif" in season or "monsoon" in season:
                return {
                    "crop": "Rice (Paddy) - Sugarcane crop cycle",
                    "yield": "22-26 quintals of Paddy per acre.",
                    "resources": "NPK 50:25:25 kg/acre, Zinc Sulfate 10kg/acre. Requires high irrigation.",
                    "rotation": "Rotate with Wheat or Mustard in winter.",
                    "tips": "Prepare nursery 25 days before transplanting. Maintain water standing of 2-5cm during early vegetative phase."
                }
            else:
                return {
                    "crop": "Wheat (Kanak / Gehun)",
                    "yield": "18-22 quintals per acre.",
                    "resources": "NPK 48:24:12 kg/acre. Seed treatment with Azotobacter.",
                    "rotation": "Rotate with Green Gram (Moong) in summer to enrich soil organic carbon.",
                    "tips": "Sow between Nov 10 - Nov 25. Requires 4 to 6 critical irrigation cycles."
                }
        else:
            # Default fallback (Red Soil/Sandy Soil)
            return {
                "crop": "Groundnut (Moongphali) or Pearl Millet (Bajra)",
                "yield": "7-10 quintals per acre.",
                "resources": "NPK 10:20:20 kg/acre. Gypsum 100kg/acre (for Groundnut shell filling).",
                "rotation": "Rotate with Pigeon Pea (Arhar/Tur).",
                "tips": "Well-drained soil is crucial. Sow after the first major shower in June."
            }

    def recommend(self, soil_type, npk, location, season, rainfall, use_ollama=False):
        n, p, k = npk
        prompt = f"""
        You are KrishiMitra's Crop Recommendation Agent.
        Farmer Details:
        - Location: {location}
        - Soil Type: {soil_type}
        - Soil Nutrients (NPK ratio): Nitrogen={n}, Phosphorus={p}, Potassium={k}
        - Current Season/Month: {season}
        - Rainfall Patterns: {rainfall}
        
        Provide:
        1. Recommended Crops to Cultivate (give 1 primary and 1 backup crop suitable for India)
        2. Expected Yield (in quintals per acre)
        3. Required Resources (seeds, fertilizers, initial water)
        4. Crop Rotation suggestions
        
        Format your response cleanly using bullet points. Answer in a professional, empathetic farming expert tone.
        """
        if use_ollama:
            try:
                return self.call_ollama(prompt, "You are a senior agronomist advising Indian farmers.")
            except ConnectionError:
                pass
        
        # Fallback
        res = self.get_fallback_recommendation(soil_type, npk, location, season, rainfall)
        return f"""**Recommended Crop:** {res['crop']}
- **Expected Yield:** {res['yield']}
- **Required Resources:** {res['resources']}
- **Crop Rotation Suggestion:** {res['rotation']}
- **Agronomist Sowing Tips:** {res['tips']}
*(Using Offline Expert Database)*"""

# 2. Pest & Disease Detection Agent
class DiseaseDetectionAgent(BaseAgent):
    def get_info(self, disease_name):
        return f"Information about {disease_name}"

# 3. Weather Agent
class WeatherAgent(BaseAgent):
    def get_weather_forecast(self, location, use_ollama=False):
        # Local forecast simulation based on major Indian cities
        loc = location.lower()
        forecast = {
            "temp": "32°C",
            "condition": "Partly Cloudy with Monsoon Winds",
            "humidity": "78%",
            "rain_prob": "75%",
            "warning": "Rain expected in the next 36 hours (Approx 20mm).",
            "action": "Delay nitrogen fertilizer top-dressing and hold irrigation for 2 days to prevent nutrient leaching. Harvest ripe crops immediately."
        }
        if "punjab" in loc or "haryana" in loc:
            forecast["temp"] = "38°C"
            forecast["condition"] = "Dry & Hot (Loo winds)"
            forecast["rain_prob"] = "5%"
            forecast["warning"] = "Heatwave alert. Temperatures exceeding 40°C expected in afternoon."
            forecast["action"] = "Schedule early morning light irrigation. Apply straw mulching around crop roots to retain soil moisture."
        elif "rajasthan" in loc:
            forecast["temp"] = "41°C"
            forecast["condition"] = "Extreme Heat / Dust Storm"
            forecast["rain_prob"] = "2%"
            forecast["warning"] = "Severe dry conditions."
            forecast["action"] = "Use drip irrigation strictly in evening. Ensure windbreaks/shelterbelts are clear of dry debris."
            
        prompt = f"""
        You are KrishiMitra's Weather Intelligence Agent.
        Analyze this forecast for {location}:
        Temperature: {forecast['temp']}, Condition: {forecast['condition']}, Humidity: {forecast['humidity']}, Rain probability: {forecast['rain_prob']}
        Special warnings: {forecast['warning']}
        
        Generate localized farming warnings and actionable instructions for the farmer (e.g. when to irrigate, spray pesticides, or sow).
        Keep the response under 100 words.
        """
        if use_ollama:
            try:
                return self.call_ollama(prompt, "You are a weather coordinator for Indian agriculture.")
            except ConnectionError:
                pass
                
        return f"""🌦 **Current Conditions ({location}):** {forecast['temp']} | {forecast['condition']}
- **Humidity:** {forecast['humidity']} | **Rain Probability:** {forecast['rain_prob']}
- **🚨 Warning:** {forecast['warning']}
- **💡 Actionable Advice:** {forecast['action']}
*(Using Offline Weather Simulator)*"""

# 4. Fertilizer & Irrigation Advisor Agent
class IrrigationAdvisorAgent(BaseAgent):
    def advise(self, crop_name, stage, soil_condition, rain_forecast, use_ollama=False):
        crop = crop_name.lower()
        stage = stage.lower()
        
        # Logic lookup
        if "rice" in crop or "paddy" in crop:
            water_need = "Maintained water depth of 3-5 cm. Dry irrigation when soil hairline cracks appear."
            fert_need = "Top dress Urea (30kg/acre) mixed with neem cake powder at tillering stage."
        elif "cotton" in crop:
            water_need = "Moderate water requirement. Drip irrigation weekly. Avoid flooding."
            fert_need = "Apply Nitrogen & Potassium split dosage (20kg N, 10kg K2O per acre) at flowering stage."
        elif "wheat" in crop:
            water_need = "Irrigate during Crown Root Initiation (CRI) stage (21 days post sowing) and Jointing stage."
            fert_need = "Apply remaining 50% Nitrogen (Urea) just before the second irrigation cycle."
        else:
            water_need = "Drip irrigation every 4-6 days based on soil moisture check."
            fert_need = "Apply balanced NPK 19:19:19 water-soluble foliar spray at vegetative stages."

        if "rain" in rain_forecast.lower() or "expected" in rain_forecast.lower():
            irrigation_instruction = "⚠️ **Urgent:** Rain is in the forecast. Stop/Delay irrigation to prevent root rot and wastage."
        else:
            irrigation_instruction = "Irrigation is safe. Proceed according to scheduled guidelines."

        prompt = f"""
        You are KrishiMitra's Fertilizer & Irrigation Advisor.
        Recommend irrigation schedule and fertilizer dosage for:
        - Crop: {crop_name}
        - Growth Stage: {stage}
        - Soil Condition: {soil_condition}
        - Weather / Rain forecast: {rain_forecast}
        
        Promote sustainable agricultural practices (like drip irrigation, bio-fertilizers, neem-coated urea). Keep it concise.
        """
        if use_ollama:
            try:
                return self.call_ollama(prompt, "You are a soil health and irrigation engineer advising Indian farmers.")
            except ConnectionError:
                pass
                
        return f"""💧 **Water Requirements ({crop_name} - {stage} stage):**
- {water_need}
- **Irrigation Status:** {irrigation_instruction}

🌱 **Fertilizer Dosage:**
- {fert_need}
- **Sustainable tip:** Incorporate Vermicompost and Bio-fertilizers (Azotobacter/PSB) to cut chemical costs by 20%.
*(Using Offline Fertilizer Expert)*"""

# 5. Market Price Intelligence Agent
class MarketIntelligenceAgent(BaseAgent):
    def get_market_trends(self, crop, state, use_ollama=False):
        crop_key = crop.lower()
        # Simulated Mandi database for Indian APMCs
        mandi_data = {
            "soybean": {
                "mandi": "Latur Mandi (Maharashtra)",
                "price": "₹4,600 - ₹4,850 per Quintal",
                "trend": "Upwards 📈 (Increased export demand for soy meal)",
                "ename": "eNAM average: ₹4,720/Quintal",
                "storage": "Store in dry gunny bags on wooden planks. Price expected to rise by ₹150 in 30 days."
            },
            "cotton": {
                "mandi": "Rajkot Mandi (Gujarat)",
                "price": "₹6,800 - ₹7,300 per Quintal",
                "trend": "Stable ➡️ (Balanced demand from spinning mills)",
                "ename": "eNAM average: ₹7,050/Quintal",
                "storage": "Keep in moisture-free storage. Cotton is prone to staining if stored on floors."
            },
            "wheat": {
                "mandi": "Khanna Mandi (Punjab)",
                "price": "₹2,275 - ₹2,350 per Quintal (MSP: ₹2,275)",
                "trend": "Slightly Upwards 📈",
                "ename": "eNAM average: ₹2,310/Quintal",
                "storage": "Treat grain with neem leaves or dry ginger powder before storage to prevent weevils."
            },
            "rice": {
                "mandi": "Burdwan Mandi (West Bengal)",
                "price": "₹2,180 - ₹2,400 per Quintal",
                "trend": "Stable ➡️",
                "ename": "eNAM average: ₹2,250/Quintal",
                "storage": "Maintain storage humidity below 12% to avoid grain discoloration."
            }
        }

        selected = mandi_data.get(crop_key, {
            "mandi": f"Local APMC (State: {state})",
            "price": "₹2,100 - ₹2,500 per Quintal (MSP benchmark)",
            "trend": "Stable ➡️",
            "ename": "eNAM average: ₹2,300/Quintal",
            "storage": "Store in ventilated godowns. Prevent rodent entry."
        })

        prompt = f"""
        You are KrishiMitra's Market Intelligence Agent.
        Provide market price details for:
        Crop: {crop}, State: {state}.
        Mandi Rate: {selected['price']} in {selected['mandi']}.
        eNAM: {selected['ename']}.
        Trend: {selected['trend']}.
        Storage: {selected['storage']}.
        
        Synthesize this into a helpful recommendation advising whether the farmer should sell now or store the crop.
        """
        if use_ollama:
            try:
                return self.call_ollama(prompt, "You are an agricultural economist monitoring APMC mandis.")
            except ConnectionError:
                pass
                
        return f"""💰 **APMC Market Price Update ({crop} - {state}):**
- **Highest Paying Mandi:** {selected['mandi']}
- **Current Mandi Rate:** {selected['price']}
- **eNAM Digital Price:** {selected['ename']}
- **Price Trend:** {selected['trend']}
- **📦 Storage Advice & Sellig strategy:** {selected['storage']}
*(Using Offline Mandi Database)*"""

# 6. Government Scheme Agent
class GovernmentSchemeAgent(BaseAgent):
    def get_schemes_info(self, scheme_query, use_ollama=False):
        schemes = {
            "pm-kisan": {
                "title": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "benefits": "₹6,000 per year paid in three equal installments of ₹2,000 directly into bank accounts.",
                "eligibility": "All landholding farmer families (subject to exclusion criteria like institutional landholders, taxpayers).",
                "process": "Register on PM-KISAN portal (pmkisan.gov.in) with Aadhaar, land mutation document, and bank passbook."
            },
            "pmfby": {
                "title": "PMFBY (Pradhan Mantri Fasal Bima Yojana)",
                "benefits": "Comprehensive crop insurance against natural calamities. Premium is heavily subsidized (2% Kharif, 1.5% Rabi, 5% Commercial).",
                "eligibility": "All farmers including sharecroppers and tenant farmers growing notified crops in notified areas.",
                "process": "Apply via CSC (Common Service Center) or online through National Crop Insurance Portal (pmfby.gov.in) within sowing timelines."
            },
            "subsidies": {
                "title": "Agricultural Machinery and Solar Pump Subsidies (PM-KUSUM)",
                "benefits": "Up to 60% subsidy for installing solar pumps and 50% subsidy for buying tractors, seed drills, or laser levellers.",
                "eligibility": "Small and marginal farmers, registered farmer co-operatives.",
                "process": "Apply on State Agriculture Department portal (e.g. MahaDBT in Maharashtra) with land 7/12 extract and quotes."
            }
        }

        query = scheme_query.lower()
        matched = schemes.get("pm-kisan") # default
        if "insurance" in query or "pmfby" in query or "crop" in query:
            matched = schemes.get("pmfby")
        elif "pump" in query or "solar" in query or "subsidy" in query or "tractor" in query or "machine" in query:
            matched = schemes.get("subsidies")

        prompt = f"""
        You are KrishiMitra's Government Scheme Assistant.
        Provide simple information on:
        Scheme: {matched['title']}
        Benefits: {matched['benefits']}
        Eligibility: {matched['eligibility']}
        How to apply: {matched['process']}
        
        Write it in a friendly, conversational, easy-to-understand layout for a village farmer.
        """
        if use_ollama:
            try:
                return self.call_ollama(prompt, "You are a rural development coordinator helping farmers.")
            except ConnectionError:
                pass
                
        return f"""🏛 **Scheme Profile: {matched['title']}**
- **🎁 Key Benefits:** {matched['benefits']}
- **👥 Eligibility Criteria:** {matched['eligibility']}
- **📝 Application Procedure:** {matched['process']}
*(Using Offline Sarkari Yojana Database)*"""


# Master Coordinator Agent
class MasterCoordinator(BaseAgent):
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3"):
        super().__init__(ollama_url, model_name)
        self.crop_agent = CropRecommendationAgent(ollama_url, model_name)
        self.weather_agent = WeatherAgent(ollama_url, model_name)
        self.irrigation_agent = IrrigationAdvisorAgent(ollama_url, model_name)
        self.market_agent = MarketIntelligenceAgent(ollama_url, model_name)
        self.scheme_agent = GovernmentSchemeAgent(ollama_url, model_name)

    def route_and_solve(self, query, use_ollama=False):
        q = query.lower()
        
        # Categorize query
        if any(w in q for w in ["crop", "grow", "sow", "plant", "cultivate", "soil", "rotation", "उगाएं", "फसल", "जमीन"]):
            # Crop Recommendation
            return self.crop_agent.recommend(
                soil_type="Black Cotton / Medium Clay",
                npk=(30, 20, 15),
                location="Central India / Maharashtra",
                season="June (Kharif Monsoon)",
                rainfall="Medium expected",
                use_ollama=use_ollama
            )
        elif any(w in q for w in ["weather", "rain", "heatwave", "temp", "cold", "forecast", "मौसम", "बारिश", "धूप"]):
            # Weather
            return self.weather_agent.get_weather_forecast("Maharashtra / Central region", use_ollama=use_ollama)
        elif any(w in q for w in ["fertilizer", "water", "irrigate", "urea", "dap", "npk", "drip", "सिंचाई", "खाद"]):
            # Irrigation & Fertilizer
            return self.irrigation_agent.advise(
                crop_name="Cotton",
                stage="Flowering stage",
                soil_condition="Humid Black Soil",
                rain_forecast="Light showers expected in 2 days",
                use_ollama=use_ollama
            )
        elif any(w in q for w in ["price", "market", "mandi", "sell", "enam", "rate", "cost", "दाम", "बाजार", "मंडी"]):
            # Market Intelligence
            return self.market_agent.get_market_trends("Soybean", "Maharashtra", use_ollama=use_ollama)
        elif any(w in q for w in ["scheme", "yojana", "pm-kisan", "subsidy", "insurance", "loan", "सरकारी", "योजना", "बीमा"]):
            # Schemes
            return self.scheme_agent.get_schemes_info(q, use_ollama=use_ollama)
        else:
            # General helper / conversational
            prompt = f"""
            You are KrishiMitra AI, the digital assistant for Indian farmers.
            Answer the farmer's query comprehensively and warmly:
            Query: "{query}"
            
            Mention that you can help with crop recommendations, weather forecasting, disease detection (via the upload tab), market mandi prices, irrigation schedules, and government schemes (Yojanas).
            Keep it clear, simple, and supportive.
            """
            if use_ollama:
                try:
                    return self.call_ollama(prompt, "You are KrishiMitra AI, an empathetic Indian agricultural assistant.")
                except ConnectionError:
                    pass
            
            return """**Namaste! I am KrishiMitra AI (कृषिमित्र).** 🙏
I am ready to help you optimize your farming yield and increase your income. You can ask me questions about:
1. 🌾 **Crop Selection:** Which crop is best for your soil, season, and rainfall.
2. 🌦 **Weather Alerts:** Current weather forecasts and irrigation advisories.
3. 🐛 **Disease Detection:** Upload pictures of crop leaf spots in the **Pest & Disease Doctor** tab for organic treatments.
4. 💧 **Fertilizer & Irrigation:** Optimized dosage and watering guidelines.
5. 💰 **Mandi Prices:** Live APMC market rates and selling strategies.
6. 🏛 **Government Schemes:** Subsidies, PM-KISAN, and crop insurance (PMFBY).

*Tip: You can use the buttons/tabs above to fill details directly!*"""
