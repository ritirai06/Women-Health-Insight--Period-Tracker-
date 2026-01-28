"""
Comprehensive Recommendation System for Women's Health
Generates personalized health insights based on cycle data, lifestyle factors, and medical conditions
"""


def get_bmi_category(bmi: float) -> str:
    """Categorize BMI into standard ranges"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def generate_personalized_recommendations(
    wellness_score: float,
    cycle_length: int,
    period_duration: int,
    sleep_hours: float,
    stress_level: str,
    exercise_frequency: str,
    water_intake: int,
    diet_quality: str,
    bmi: float,
    cramp_severity: int,
    has_pcos: bool,
    has_endometriosis: bool,
    has_thyroid: bool,
    mood_state: str,
    symptoms: list,
    pred_days: float,
    age: int = 25,
    contraceptive_use: str = "none"
) -> list:
    """
    Generate AI-powered personalized health recommendations with enhanced insights
    
    Returns:
        List of recommendation dictionaries with category, title, advice, action, and priority
    """
    recommendations = []
    
    # ═══════════════════════════════════════════════════════════════
    # 🚨 CRITICAL: Hormonal Imbalance / Skipped Periods Alert
    # ═══════════════════════════════════════════════════════════════
    if pred_days >= 60:  # 2+ months delay
        months_delayed = pred_days / 30
        recommendations.append({
            "category": "🚨 URGENT: Medical Attention Required",
            "title": f"Severe Delay Detected - {months_delayed:.1f} Month(s) Skipped Period",
            "advice": f"A delay of {pred_days:.0f} days ({months_delayed:.1f} months) indicates a SKIPPED PERIOD and potential hormonal imbalance. This requires immediate medical evaluation. Possible causes include PCOS, thyroid disorders, pregnancy, premature ovarian insufficiency, or significant hormonal disruption.",
            "action": "⚠️ SCHEDULE URGENT APPOINTMENT with gynecologist/endocrinologist within 48-72 hours. Get comprehensive hormonal panel: FSH, LH, Estradiol, Progesterone, TSH, Prolactin, Testosterone, AMH. Rule out pregnancy first.",
            "priority": "CRITICAL"
        })
    elif pred_days >= 45:  # 1.5 months delay
        recommendations.append({
            "category": "🔴 High Priority Medical Review",
            "title": f"Significant Irregularity - {pred_days:.0f} Day Delay",
            "advice": "This level of delay suggests oligomenorrhea (infrequent periods) and requires medical investigation. Your hormonal balance may be significantly disrupted.",
            "action": "Schedule appointment with healthcare provider within 1 week. Request hormonal blood work and pelvic ultrasound to assess ovarian function and rule out cysts or other abnormalities.",
            "priority": "HIGH"
        })
    elif pred_days >= 15:
        recommendations.append({
            "category": "🟡 Moderate Concern",
            "title": "Irregular Cycle Pattern Detected",
            "advice": f"A {pred_days:.0f}-day delay indicates menstrual irregularity. This could be due to stress, lifestyle factors, or early signs of hormonal imbalance.",
            "action": "Monitor for 2-3 cycles. If pattern persists, consult healthcare provider. Track basal body temperature to confirm ovulation.",
            "priority": "MODERATE"
        })
    elif pred_days >= 7:
        recommendations.append({
            "category": "🟢 Minor Irregularity",
            "title": "Slight Cycle Variation",
            "advice": "A 7-14 day delay is common and often related to stress, travel, or lifestyle changes. Monitor for patterns.",
            "action": "Track cycle for next 3 months. Focus on stress management and consistent sleep schedule.",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 😴 SLEEP HEALTH
    # ═══════════════════════════════════════════════════════════════
    if sleep_hours < 6:
        recommendations.append({
            "category": "😴 Sleep Health - Critical",
            "title": "Severe Sleep Deprivation",
            "advice": f"You're averaging only {sleep_hours:.1f} hours of sleep. This severely disrupts cortisol and melatonin production, directly affecting reproductive hormones (estrogen, progesterone). Chronic sleep deprivation can cause amenorrhea (absent periods) and anovulation.",
            "action": "🎯 PRIORITY ACTION: Aim for 7.5-8 hours minimum. Create non-negotiable sleep schedule. Avoid screens 2 hours before bed. Consider sleep study if you have insomnia. Supplement with magnesium glycinate (400mg) before bed.",
            "priority": "HIGH"
        })
    elif sleep_hours < 7:
        recommendations.append({
            "category": "😴 Sleep Health",
            "title": "Insufficient Sleep Pattern",
            "advice": f"You're averaging {sleep_hours:.1f} hours of sleep. Aim for 7-9 hours nightly. Poor sleep can disrupt hormonal balance, increase cortisol, and affect cycle regularity.",
            "action": "Establish a consistent bedtime routine. Use blackout curtains, keep room cool (65-68°F). Avoid caffeine after 2 PM. Try 3-6mg melatonin 30 minutes before bed if needed.",
            "priority": "MODERATE"
        })
    elif sleep_hours > 10:
        recommendations.append({
            "category": "😴 Sleep Health",
            "title": "Excessive Sleep Duration",
            "advice": f"Sleeping {sleep_hours:.1f} hours regularly may indicate underlying fatigue, depression, thyroid issues, or vitamin deficiencies (especially B12, D, iron).",
            "action": "Get blood work: Complete Blood Count (CBC), Vitamin D, B12, ferritin, thyroid panel (TSH, T3, T4). Rule out sleep apnea or depression. Set consistent wake-up time.",
            "priority": "MODERATE"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 🧘 STRESS MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    if stress_level == "high":
        recommendations.append({
            "category": "🧘 Stress Management - Critical",
            "title": "Elevated Stress Levels - Hormonal Impact",
            "advice": "High chronic stress elevates cortisol, which directly suppresses GnRH (gonadotropin-releasing hormone) production. This cascades to reduced FSH/LH, leading to anovulation and irregular/absent periods. Stress is a MAJOR cause of menstrual irregularity.",
            "action": "⚠️ IMMEDIATE: Daily stress reduction is non-negotiable. Practice: (1) 10-15 min meditation daily (Headspace/Calm apps), (2) Yoga or tai chi 3x/week, (3) Deep breathing exercises (4-7-8 technique), (4) Consider adaptogenic herbs (Ashwagandha 300-500mg daily), (5) Therapy/counseling if chronic stress, (6) Magnesium L-threonate supplement (2000mg).",
            "priority": "CRITICAL"
        })
    elif stress_level == "medium":
        recommendations.append({
            "category": "🧘 Stress Management",
            "title": "Moderate Stress Levels",
            "advice": "Moderate stress can still impact hormonal balance. Cortisol affects the hypothalamic-pituitary-ovarian (HPO) axis regulating your menstrual cycle.",
            "action": "Practice daily mindfulness (10 min). Try progressive muscle relaxation. Ensure work-life balance. Journaling can help identify stress triggers. Consider B-complex vitamins for stress support.",
            "priority": "MODERATE"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 💪 EXERCISE & PHYSICAL ACTIVITY
    # ═══════════════════════════════════════════════════════════════
    if exercise_frequency == "sedentary":
        recommendations.append({
            "category": "💪 Physical Activity - Important",
            "title": "Sedentary Lifestyle - Hormonal Consequences",
            "advice": "Physical inactivity increases insulin resistance, promotes weight gain (especially visceral fat), and disrupts sex hormone binding globulin (SHBG) levels. This leads to hormonal imbalances, particularly elevated androgens (testosterone) and estrogen dominance, contributing to irregular cycles.",
            "action": "🎯 START GRADUALLY: Week 1-2: 15-min walks daily. Week 3-4: 30-min walks 5x/week. Month 2+: Add strength training 2x/week. Goal: 150 min moderate exercise weekly. Exercise improves insulin sensitivity and hormonal balance within 8-12 weeks.",
            "priority": "HIGH"
        })
    elif "active" in exercise_frequency.lower() and bmi < 18.5:
        recommendations.append({
            "category": "💪 Physical Activity - Overtraining Risk",
            "title": "Exercise Intensity + Low BMI = Hypothalamic Amenorrhea Risk",
            "advice": f"Excessive exercise combined with low body weight (BMI {bmi:.1f}) can cause Functional Hypothalamic Amenorrhea (FHA) or 'Athletic Amenorrhea'. Low body fat reduces leptin, which signals the brain to shut down reproductive function to conserve energy.",
            "action": "⚠️ REDUCE exercise to 3-4 days/week maximum. Eliminate high-intensity workouts. Focus on yoga, walking, swimming. INCREASE caloric intake by 300-500 calories daily, especially healthy fats (avocado, nuts, olive oil). May need to gain 5-10 lbs to restore cycles. Consider working with sports nutritionist.",
            "priority": "CRITICAL"
        })
    elif "light" in exercise_frequency.lower():
        recommendations.append({
            "category": "💪 Physical Activity",
            "title": "Light Activity - Room for Improvement",
            "advice": "Light exercise is good, but increasing to moderate intensity can significantly improve insulin sensitivity and hormonal balance.",
            "action": "Gradually increase to 30-40 minutes moderate intensity 4-5x/week. Include both cardio and strength training. Strength training particularly improves PCOS symptoms.",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 💧 HYDRATION
    # ═══════════════════════════════════════════════════════════════
    if water_intake < 6:
        recommendations.append({
            "category": "💧 Hydration - Undervalued Factor",
            "title": "Chronic Dehydration Effects",
            "advice": f"Only {water_intake} glasses daily is insufficient. Dehydration increases cortisol production, impairs detoxification of excess hormones through the liver and kidneys, and worsens PMS symptoms (bloating, fatigue, headaches). Proper hydration supports optimal endocrine function.",
            "action": "🎯 TARGET: 8-10 glasses (64-80 oz) daily. STRATEGY: (1) Drink 16 oz upon waking, (2) 8 oz before each meal, (3) Keep water bottle visible, (4) Set hourly phone reminders, (5) Herbal teas count (avoid caffeine). Track intake for 1 week to build habit.",
            "priority": "MODERATE"
        })
    elif water_intake >= 10:
        recommendations.append({
            "category": "💧 Hydration",
            "title": "Excellent Hydration!",
            "advice": f"Great job maintaining {water_intake} glasses daily! This supports hormonal detoxification and reduces inflammation.",
            "action": "Continue current habits. Ensure electrolyte balance if drinking >12 glasses daily (add pinch of sea salt or electrolyte powder).",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 🥗 NUTRITION
    # ═══════════════════════════════════════════════════════════════
    if diet_quality in ["poor", "fair"]:
        recommendations.append({
            "category": "🥗 Nutrition - Foundation of Hormonal Health",
            "title": "Dietary Improvement Crucial for Cycle Regulation",
            "advice": "Poor nutrition is a PRIMARY cause of menstrual irregularities. Specific nutrient deficiencies directly impact hormone production: Vitamin B6 (progesterone synthesis), Magnesium (300+ enzymes including hormone metabolism), Zinc (ovulation), Iron (energy + hemoglobin), Omega-3s (anti-inflammatory, prostaglandin regulation).",
            "action": "🎯 ESSENTIAL FOODS: Daily - (1) Leafy greens (kale, spinach) - folate, magnesium, iron, (2) Fatty fish (salmon, sardines) 3x/week - omega-3s, vitamin D, (3) Nuts/seeds (pumpkin seeds, almonds) - zinc, magnesium, healthy fats, (4) Whole grains (quinoa, oats) - B vitamins, fiber, (5) Legumes (lentils, chickpeas) - iron, protein, fiber. AVOID: refined sugar, trans fats, excessive alcohol. Consider supplements: Multivitamin, Omega-3 (1000mg EPA/DHA), Magnesium glycinate (400mg), Vitamin D3 (2000-4000 IU).",
            "priority": "CRITICAL"
        })
    elif diet_quality == "good":
        recommendations.append({
            "category": "🥗 Nutrition",
            "title": "Good Diet - Optimize Further",
            "advice": "You're eating well! Fine-tune for cycle support: focus on cruciferous vegetables (broccoli, cauliflower) for estrogen metabolism, and seed cycling (pumpkin/flax in follicular phase, sesame/sunflower in luteal phase).",
            "action": "Add 1-2 tbsp ground flaxseed daily (supports estrogen balance). Include probiotic-rich foods (yogurt, kimchi, sauerkraut) for gut-hormone axis.",
            "priority": "LOW"
        })
    elif diet_quality == "excellent":
        recommendations.append({
            "category": "🥗 Nutrition",
            "title": "Excellent Nutrition!",
            "advice": "Outstanding dietary habits! You're providing your body with optimal nutrients for hormonal balance.",
            "action": "Maintain current eating patterns. Consider working with a functional nutritionist for personalized optimization if still experiencing cycle issues.",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # ⚖️ BMI & WEIGHT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    bmi_category = get_bmi_category(bmi)
    
    if bmi_category == "Underweight":
        recommendations.append({
            "category": "⚖️ Weight Management - Critical for Fertility",
            "title": f"Underweight (BMI {bmi:.1f}) - Amenorrhea Risk",
            "advice": "BMI below 18.5 often causes Hypothalamic Amenorrhea due to insufficient body fat. Fat tissue (adipose) produces leptin and aromatase enzyme (converts androgens to estrogen). Low body fat = low leptin = brain shuts down GnRH → no ovulation → absent/irregular periods. This is your body's survival mechanism.",
            "action": "⚠️ WEIGHT GAIN ESSENTIAL: Target BMI 20-22 for optimal fertility. Gain 0.5-1 lb/week (not faster). INCREASE calories by 500-750 daily through nutrient-dense foods (not junk). Focus: healthy fats (nuts, avocado, olive oil), complex carbs (sweet potato, oats), lean protein. Consider working with dietitian specialized in hypothalamic amenorrhea. Periods typically return at BMI 19-20, but full ovulation may need BMI 21+. Timeline: 3-6 months.",
            "priority": "CRITICAL"
        })
    elif bmi_category == "Overweight":
        recommendations.append({
            "category": "⚖️ Weight Management",
            "title": f"Overweight (BMI {bmi:.1f}) - Insulin Resistance Risk",
            "advice": "Excess weight, especially visceral (belly) fat, increases insulin resistance and androgen production (testosterone). Fat tissue produces excess estrogen through aromatase enzyme, creating estrogen dominance. This combination disrupts normal ovulation and causes irregular/absent periods. Higher risk for PCOS, endometrial hyperplasia.",
            "action": "🎯 TARGET: 5-10% weight loss can dramatically improve cycle regularity (usually within 3 months). APPROACH: (1) Reduce refined carbs/sugar - improves insulin, (2) Increase protein (20-30g per meal) - improves satiety + metabolism, (3) Strength training 3x/week - builds muscle which burns more calories, (4) Aim for 0.5-1 lb loss/week (sustainable), (5) Consider low-glycemic or Mediterranean diet. Monitor cycle changes monthly.",
            "priority": "HIGH"
        })
    elif bmi_category == "Obese":
        recommendations.append({
            "category": "⚖️ Weight Management - Urgent",
            "title": f"Obesity (BMI {bmi:.1f}) - Significant Hormonal Disruption",
            "advice": "Obesity significantly increases risk of: (1) Chronic anovulation (no ovulation), (2) PCOS development/worsening, (3) Insulin resistance → Type 2 diabetes, (4) Endometrial cancer (from unopposed estrogen), (5) Cardiovascular disease. Excess fat tissue acts as an endocrine organ producing inflammatory cytokines and hormones that severely disrupt the HPO axis.",
            "action": "⚠️ COMPREHENSIVE APPROACH NEEDED: (1) Medical supervision - consider endocrinologist + dietitian, (2) Screen for metabolic syndrome: fasting glucose, HbA1c, lipid panel, liver function, (3) Weight loss goal: 10-15% over 6 months (major cycle improvement), (4) Consider metformin if insulin resistant, (5) Diet: Low-carb or ketogenic may be most effective, eliminate sugar/processed foods, (6) Exercise: Start with 20 min walking daily, build to 45+ min 5x/week + strength training, (7) Address emotional eating/trauma if present. Even 5-7% weight loss often restores ovulation.",
            "priority": "CRITICAL"
        })
    elif bmi_category == "Normal weight":
        recommendations.append({
            "category": "⚖️ Weight Management",
            "title": f"Healthy Weight (BMI {bmi:.1f})",
            "advice": "Excellent! Maintaining a healthy BMI supports optimal reproductive hormone levels and cycle regularity.",
            "action": "Continue current weight maintenance habits. Focus on body composition (muscle vs. fat) rather than just weight.",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 💊 PAIN MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    if cramp_severity >= 7:
        recommendations.append({
            "category": "💊 Pain Management - Medical Evaluation Needed",
            "title": "Severe Dysmenorrhea (Pain ≥7/10) - NOT NORMAL",
            "advice": f"Severe pain ({cramp_severity}/10) is NOT 'just bad periods'. This level of dysmenorrhea strongly suggests: (1) Endometriosis (most likely - affects 10% of women), (2) Adenomyosis, (3) Uterine fibroids, (4) Pelvic inflammatory disease, (5) Ovarian cysts. Severe pain results from excessive prostaglandin F2α production causing intense uterine contractions, or from endometrial tissue outside the uterus (endometriosis) causing inflammation.",
            "action": "🚨 MEDICAL WORKUP REQUIRED: Schedule gynecology appointment ASAP. Request: (1) Pelvic ultrasound (transvaginal), (2) Possibly MRI if endometriosis suspected, (3) May need diagnostic laparoscopy for definitive endometriosis diagnosis. MEANWHILE: NSAIDs (ibuprofen 400-800mg every 6-8 hours) - blocks prostaglandin synthesis, start 1-2 days BEFORE period, heating pad, magnesium supplements (400mg daily), consider hormonal birth control for suppression, pelvic floor physical therapy.",
            "priority": "CRITICAL"
        })
    elif cramp_severity >= 5:
        recommendations.append({
            "category": "💊 Pain Management",
            "title": f"Moderate-Severe Cramps ({cramp_severity}/10)",
            "advice": "Moderate-high pain suggests elevated prostaglandin levels. While common, it can be significantly reduced through diet, supplements, and lifestyle.",
            "action": "NATURAL MANAGEMENT: (1) Magnesium glycinate 400mg daily (reduces prostaglandins + uterine contractions), (2) Omega-3 fish oil 2000mg daily (anti-inflammatory, reduces prostaglandin F2α), (3) Vitamin E 400 IU (reduces menstrual pain), (4) Ginger tea (natural COX-2 inhibitor), (5) Chamomile tea (antispasmodic), (6) Heat therapy (increases blood flow), (7) Gentle yoga/stretching, (8) Acupuncture (proven effective). MEDICATION: Start NSAIDs (ibuprofen/naproxen) 1-2 days BEFORE period begins for maximum effect. If no improvement in 3 months, see gynecologist.",
            "priority": "MODERATE"
        })
    elif cramp_severity >= 3:
        recommendations.append({
            "category": "💊 Pain Management",
            "title": f"Mild-Moderate Cramps ({cramp_severity}/10)",
            "advice": "Mild cramping is normal, caused by prostaglandins triggering uterine contractions to shed the endometrial lining.",
            "action": "Try natural remedies: magnesium (300mg daily), heat therapy, gentle exercise, ginger or chamomile tea. Reduce inflammatory foods (sugar, dairy, red meat) during menstruation.",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 🏥 MEDICAL CONDITIONS
    # ═══════════════════════════════════════════════════════════════
    if has_pcos:
        recommendations.append({
            "category": "🏥 PCOS Management - Comprehensive Approach",
            "title": "Polycystic Ovary Syndrome - Lifestyle is First-Line Treatment",
            "advice": "PCOS is a metabolic and hormonal disorder characterized by insulin resistance (root cause in 70% of cases), elevated androgens (testosterone), and irregular/absent ovulation. It's the #1 cause of irregular periods and infertility, affecting 10-15% of women. PCOS significantly increases long-term risks: Type 2 diabetes (50% by age 40), cardiovascular disease, endometrial cancer, fatty liver disease.",
            "action": "🎯 COMPREHENSIVE PCOS PROTOCOL: (1) DIET: Low-glycemic index, reduce refined carbs/sugar (improves insulin), increase protein + fiber, consider Mediterranean or DASH diet. (2) EXERCISE: 150 min moderate cardio + strength training 3x/week (improves insulin sensitivity within 8 weeks). (3) SUPPLEMENTS: Inositol (myo-inositol 2000mg + d-chiro-inositol 50mg daily) - improves insulin + ovulation 80% effective, Berberine 500mg 3x/day OR Metformin 1500-2000mg daily (insulin sensitizer), NAC 600mg 2x/day (improves ovulation), Vitamin D3 (4000 IU - many PCOS women deficient), Omega-3, Magnesium, Spearmint tea (anti-androgen). (4) WEIGHT LOSS: If overweight, 5-10% loss restores ovulation in 75%. (5) STRESS MANAGEMENT: Crucial - stress worsens insulin resistance. (6) MONITOR: Track cycles, check HbA1c annually, liver enzymes. (7) MEDICATIONS: Consider hormonal birth control (regulates cycles), anti-androgens (spironolactone for hirsutism/acne), ovulation induction (clomid/letrozole) if trying to conceive. Work with endocrinologist or PCOS specialist.",
            "priority": "CRITICAL"
        })
    
    if has_endometriosis:
        recommendations.append({
            "category": "🏥 Endometriosis Management - Reduce Inflammation",
            "title": "Endometriosis Care - Anti-Inflammatory Approach",
            "advice": "Endometriosis is a chronic inflammatory condition where endometrial tissue grows outside the uterus, responding to hormonal cycles and causing severe pain, inflammation, scarring, and infertility. It's estrogen-dependent and prostaglandin-driven. No cure exists, but symptoms can be managed.",
            "action": "🎯 ENDOMETRIOSIS PROTOCOL: (1) ANTI-INFLAMMATORY DIET: Eliminate or reduce: gluten, dairy, red meat, processed foods, caffeine, alcohol (all increase inflammation/prostaglandins). Increase: omega-3 rich fish, leafy greens, berries, turmeric, ginger, green tea. (2) SUPPLEMENTS: Omega-3 fish oil (2000-3000mg EPA/DHA), Curcumin/Turmeric (1000mg with black pepper), NAC 600mg 2x/day (reduces lesion size), Vitamin D3 (5000 IU), Magnesium, Probiotics (gut health crucial). (3) HORMONAL SUPPRESSION: Often need continuous birth control (skip placebo week) to suppress endometrial growth, or Mirena IUD, or GnRH agonists (Lupron) for severe cases. (4) PAIN MANAGEMENT: NSAIDs, heat therapy, pelvic floor physical therapy (essential!), acupuncture, TENS unit. (5) EXERCISE: Gentle yoga, swimming, walking (avoid high-impact). (6) STRESS REDUCTION: Meditation, therapy. (7) SURGERY: Laparoscopic excision by skilled surgeon if severe, but recurrence common. Join support groups. Work with endometriosis specialist, not just general gyn.",
            "priority": "CRITICAL"
        })
    
    if has_thyroid:
        recommendations.append({
            "category": "🏥 Thyroid Health - Critical for Menstrual Regularity",
            "title": "Thyroid Disorder - Direct Impact on Reproductive Hormones",
            "advice": "Thyroid hormones (T3, T4) directly regulate metabolism and interact with reproductive hormones. HYPOTHYROIDISM (low thyroid) causes: heavy periods, irregular cycles, anovulation, infertility, miscarriage risk. HYPERTHYROIDISM (high thyroid) causes: light/absent periods, irregular cycles. Thyroid affects sex hormone binding globulin (SHBG), prolactin, and directly impacts ovarian function.",
            "action": "🎯 THYROID OPTIMIZATION: (1) MEDICATION COMPLIANCE: Take thyroid medication (levothyroxine) on empty stomach, same time daily, 30-60 min before food. Don't take with calcium, iron, or coffee (reduces absorption). (2) MONITORING: TSH every 6-8 weeks when adjusting dose, then every 6-12 months when stable. OPTIMAL TSH for fertility: 0.5-2.5 mIU/L (not just 'in range' 0.5-5). Request FULL panel: TSH, Free T4, Free T3, TPO antibodies (Hashimoto's), Reverse T3. (3) NUTRIENTS: Selenium 200mcg (thyroid hormone conversion), Zinc, Iron (check ferritin - needs to be >50 for optimal thyroid), Iodine (if deficient, but careful with Hashimoto's), Vitamin D, B vitamins. (4) AVOID: Soy (goitrogen), excessive raw cruciferous vegetables if hypothyroid, fluoride/chlorine in water (compete with iodine). (5) STRESS: Reduces T4→T3 conversion. (6) GUT HEALTH: 80% of immune system in gut - crucial for Hashimoto's. Work with endocrinologist. Cycles should normalize within 2-3 months of optimal thyroid hormone levels.",
            "priority": "CRITICAL"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 🧠 MENTAL HEALTH
    # ═══════════════════════════════════════════════════════════════
    if mood_state in ["low", "anxious", "depressed"]:
        severity = "severe" if mood_state == "depressed" else "moderate" if mood_state == "low" else "mild"
        recommendations.append({
            "category": "🧠 Mental Wellness - Bidirectional Hormone-Mood Connection",
            "title": f"Emotional Health - {mood_state.capitalize()} Mood",
            "advice": "Hormones and mental health have bidirectional relationship: (1) Reproductive hormones (estrogen, progesterone) affect neurotransmitters (serotonin, dopamine, GABA) causing mood changes, (2) Chronic stress/anxiety/depression elevate cortisol which suppresses reproductive axis → irregular periods. Estrogen withdrawal (PMS/PMDD) can cause severe mood symptoms. Additionally, thyroid disorders often cause depression/anxiety.",
            "action": f"🎯 MENTAL HEALTH PROTOCOL: (1) PROFESSIONAL HELP: {'URGENT - See psychiatrist/therapist immediately if suicidal thoughts' if mood_state == 'depressed' else 'Schedule therapy - CBT proven effective. Consider psychiatrist if severe.'} (2) MEDICATION: May need antidepressants (SSRIs also treat PMDD), anxiolytics. Birth control can help or worsen mood - track carefully. (3) SUPPLEMENTS: Omega-3 (2000mg EPA/DHA - proven anti-depressant effect), Vitamin D3 (check levels - aim 50-80 ng/mL), B-complex (especially B6, folate, B12 - involved in neurotransmitter synthesis), Magnesium glycinate (400mg - reduces anxiety), SAMe 400-800mg (depression), L-theanine (anxiety), Rhodiola/Ashwagandha (adaptogenic). (4) LIFESTYLE: Exercise 30+ min 5x/week (as effective as antidepressants for mild-moderate depression), sunlight exposure, social connection, sleep hygiene, limit alcohol. (5) TRACK CYCLES: Note if mood worsens specific cycle days (may be PMDD - treat with SSRIs luteal phase or continuous birth control). (6) THERAPY: CBT, DBT, mindfulness-based stress reduction. (7) RULE OUT: Thyroid disorder, vitamin deficiencies (D, B12, folate, iron), hormonal birth control side effects. Don't ignore mental health - it's as important as physical health.",
            "priority": "CRITICAL" if mood_state == "depressed" else "HIGH"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 💫 SYMPTOM-SPECIFIC ADVICE
    # ═══════════════════════════════════════════════════════════════
    if symptoms and isinstance(symptoms, list):
        symptoms_lower = [s.lower() for s in symptoms]
        
        if "bloating" in symptoms_lower:
            recommendations.append({
                "category": "💫 Symptom Relief - Bloating",
                "title": "Reduce Hormonal Bloating",
                "advice": "Menstrual bloating caused by: (1) Progesterone slows digestion, (2) Estrogen causes water retention, (3) Prostaglandins affect GI tract, (4) Dietary triggers worsen symptoms.",
                "action": "REDUCE BLOATING: (1) Limit salt (1500mg/day during luteal phase), (2) Avoid carbonated drinks, (3) Reduce FODMAPs if sensitive (onions, garlic, beans, dairy, wheat), (4) Probiotics daily (especially Lactobacillus), (5) Digestive enzymes with meals, (6) Magnesium (natural muscle relaxant), (7) Herbal teas: peppermint, fennel, ginger (carminative - reduce gas), (8) Stay hydrated (counterintuitive but reduces water retention), (9) Gentle movement (walking, yoga), (10) Diuretic foods: cucumber, asparagus, watermelon, (11) Avoid excess caffeine/alcohol. Keep food diary to identify triggers.",
                "priority": "LOW"
            })
        
        if "headache" in symptoms_lower or any("migraine" in s.lower() for s in symptoms):
            recommendations.append({
                "category": "💫 Symptom Relief - Menstrual Migraines",
                "title": "Manage Hormonal Headaches",
                "advice": "Menstrual migraines triggered by estrogen withdrawal (drop before period). Affects 60% of female migraine sufferers. Estrogen affects serotonin, blood vessel dilation, and pain pathways.",
                "action": "MIGRAINE MANAGEMENT: (1) PREVENTION: Magnesium 400-600mg daily (proven prophylaxis), Riboflavin (B2) 400mg daily, CoQ10 100-300mg daily, Feverfew herb, consistent sleep, avoid trigger foods (aged cheese, chocolate, alcohol, MSG). (2) ACUTE TREATMENT: NSAIDs (start 2 days before period), triptans (sumatriptan), gepants (new class), ice packs, dark quiet room, caffeine (100mg can help early). (3) HORMONAL: Consider continuous birth control (avoid estrogen withdrawal), estrogen supplementation during menstruation, Mirena IUD. (4) TRACK: Use migraine diary app - identify patterns. (5) HYDRATION: Dehydration common trigger. (6) SPECIALIST: See neurologist if severe/frequent (>4/month). Avoid medication overuse headaches. Botox or CGRP inhibitors if chronic.",
                "priority": "MODERATE"
            })
        
        if "fatigue" in symptoms_lower:
            recommendations.append({
                "category": "💫 Symptom Relief - Menstrual Fatigue",
                "title": "Combat Period-Related Exhaustion",
                "advice": "Menstrual fatigue caused by: (1) Iron loss from bleeding → iron deficiency anemia (most common), (2) Progesterone metabolites have sedative effects, (3) Inflammatory prostaglandins, (4) Poor sleep from pain/discomfort, (5) Vitamin B12/folate deficiency, (6) Thyroid issues.",
                "action": "ADDRESS FATIGUE: (1) BLOOD WORK: CBC (check hemoglobin, hematocrit), Ferritin (should be >50 ng/mL, ideally 70-90 for optimal energy - low normal not enough!), Iron panel, B12, Folate, Vitamin D, TSH. (2) IRON SUPPLEMENTATION: If ferritin <50: Take iron bisglycinate 25-50mg daily with vitamin C (enhances absorption), avoid with coffee/tea (reduces absorption), take on empty stomach if tolerated, expect 8-12 weeks to restore levels. FOOD SOURCES: Red meat (heme iron - best absorbed), liver (iron-rich), spinach, lentils, fortified cereals (non-heme - take with vitamin C). (3) B12: Sublingual methylcobalamin 1000mcg daily if low, or weekly injections. (4) VITAMIN D: 4000 IU daily if deficient (common in fatigue). (5) REDUCE BLEEDING: Tranexamic acid, birth control, Mirena IUD if heavy periods. (6) LIFESTYLE: Prioritize sleep (7-9 hours), adaptogenic herbs (Rhodiola, Cordyceps, Ashwagandha), reduce sugar (causes energy crashes), balance blood sugar (protein with each meal), gentle exercise (increases energy paradoxically). (7) RULE OUT: Thyroid disorder, sleep apnea, chronic fatigue syndrome, depression. Track energy levels with menstrual cycle to identify patterns.",
                "priority": "HIGH"
            })
    
    # ═══════════════════════════════════════════════════════════════
    # 🔄 CYCLE LENGTH ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    if cycle_length < 24:
        recommendations.append({
            "category": "🔄 Cycle Health - Short Cycles",
            "title": "Polymenorrhea - Frequent Periods",
            "advice": f"Cycles shorter than 24 days ({cycle_length} days) indicate polymenorrhea. Possible causes: (1) Luteal phase defect (insufficient progesterone production post-ovulation → early period), (2) Anovulatory cycles (no ovulation → irregular shedding), (3) Perimenopause (if age 40+, ovarian reserve declining), (4) Thyroid disorders (especially hyperthyroidism), (5) Polyps or fibroids, (6) High stress, (7) Extreme exercise/low body weight.",
            "action": "EVALUATION NEEDED: (1) Track basal body temperature (BBT) daily - biphasic pattern confirms ovulation + shows luteal phase length (should be 12-14 days), (2) Day 21 progesterone blood test (confirms ovulation + adequacy), (3) Thyroid panel (TSH, Free T4, Free T3), (4) Prolactin level (high prolactin shortens cycle), (5) Pelvic ultrasound (rule out structural issues), (6) FSH + Estradiol day 3 testing (ovarian reserve if age 35+). TREATMENT depends on cause: Progesterone supplementation if luteal phase defect, thyroid treatment if abnormal, lifestyle modifications. Frequent periods can cause iron deficiency anemia - check ferritin. See gynecologist for workup.",
            "priority": "MODERATE"
        })
    elif cycle_length > 35:
        recommendations.append({
            "category": "🔄 Cycle Health - Long Cycles",
            "title": "Oligomenorrhea - Infrequent Periods",
            "advice": f"Cycles longer than 35 days ({cycle_length} days) indicate oligomenorrhea. Possible causes: (1) PCOS (most common - 70% of oligomenorrhea), (2) Hypothalamic amenorrhea (stress, low weight, excessive exercise), (3) Thyroid disorders (especially hypothyroidism), (4) Hyperprolactinemia, (5) Perimenopause/premature ovarian insufficiency, (6) Anovulation (not ovulating regularly).",
            "action": "COMPREHENSIVE WORKUP: (1) Hormonal panel: Day 2-5 testing: FSH, LH (elevated LH:FSH ratio suggests PCOS), Estradiol, Testosterone (total + free), DHEA-S, 17-hydroxyprogesterone, Prolactin, TSH, Free T4. (2) Pelvic ultrasound: Check for polycystic ovaries (12+ follicles per ovary), endometrial thickness. (3) Progesterone challenge: Take progesterone 10 days - if period comes, means sufficient estrogen (likely PCOS). If no period, suggests very low estrogen (hypothalamic amenorrhea or POI). (4) Track ovulation: Use ovulation predictor kits (OPKs) or BBT charting. TREATMENT depends on diagnosis: Lifestyle changes for PCOS/hypothalamic amenorrhea, thyroid medication, dopamine agonist (bromocriptine) for hyperprolactinemia. Long cycles increase endometrial cancer risk from unopposed estrogen - may need progesterone withdrawal bleeds or continuous birth control. Work with reproductive endocrinologist.",
            "priority": "HIGH"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 💊 CONTRACEPTIVE CONSIDERATIONS
    # ═══════════════════════════════════════════════════════════════
    if contraceptive_use and contraceptive_use.lower() != "none":
        if "oral" in contraceptive_use.lower():
            recommendations.append({
                "category": "💊 Contraceptive Considerations",
                "title": "Birth Control Pills - Effects on Cycle",
                "advice": "Oral contraceptives suppress natural hormone production and ovulation. The 'period' on pills is actually withdrawal bleeding, not a true menstrual period. Pills regulate cycles artificially but mask underlying issues.",
                "action": "If experiencing irregular bleeding on pills: (1) Take at same time daily (timing crucial), (2) May need higher estrogen dose or different progestin formulation, (3) Check for drug interactions (antibiotics, anti-seizure meds reduce effectiveness), (4) Consider other methods if problematic. Be aware: Coming off pills may reveal underlying cycle issues (PCOS, etc.). May take 3-6 months for natural cycles to return. Supplement with B vitamins (pills deplete), folate, magnesium.",
                "priority": "LOW"
            })
        elif "iud" in contraceptive_use.lower():
            recommendations.append({
                "category": "💊 Contraceptive Considerations",
                "title": "IUD - Cycle Changes Expected",
                "advice": "Mirena (hormonal IUD): Often causes light periods or amenorrhea (no period) - this is normal and safe, not a concern. Copper IUD: Often causes heavier, more painful periods.",
                "action": "Mirena: If concerned about amenorrhea, this is intentional effect - safe and doesn't indicate problem. Enjoy not having periods! Copper IUD: If periods too heavy/painful, consider tranexamic acid, NSAIDs, or switching to hormonal method. Both: Report severe pain, fever, abnormal discharge (possible infection/perforation).",
                "priority": "LOW"
            })
    
    # ═══════════════════════════════════════════════════════════════
    # ⭐ OVERALL WELLNESS
    # ═══════════════════════════════════════════════════════════════
    if wellness_score < 50:
        recommendations.append({
            "category": "⭐ Overall Wellness - Critical Status",
            "title": "Comprehensive Health Overhaul Needed",
            "advice": f"Your wellness score of {wellness_score:.0f} indicates MULTIPLE high-priority areas requiring immediate attention. Low wellness score correlates with poor cycle regularity, increased disease risk, and reduced quality of life. You deserve better health!",
            "action": "🚨 30-DAY HEALTH RESET: Week 1: (1) Schedule medical appointments (gynecologist, primary care - get full blood work), (2) Start sleep routine (same bedtime, 7-8 hours), (3) Begin daily 15-min walk. Week 2: (4) Improve diet - eliminate processed foods, add vegetables to each meal, (5) Start stress management - 10 min meditation daily, (6) Increase water to 8 glasses. Week 3: (7) Start supplements: Multivitamin, Omega-3, Vitamin D, Magnesium, (8) Increase exercise to 30 min 5x/week, (9) Meal prep healthy foods. Week 4: (10) Address mental health - therapy if needed, (11) Join support group, (12) Track all habits + cycle. Focus on ONE habit at a time. Small consistent changes create transformation. Consider health coach or functional medicine doctor for personalized guidance. YOU CAN DO THIS!",
            "priority": "CRITICAL"
        })
    elif wellness_score < 60:
        recommendations.append({
            "category": "⭐ Overall Wellness - Needs Improvement",
            "title": "Multiple Health Areas Require Attention",
            "advice": f"Wellness score of {wellness_score:.0f} indicates several areas need improvement. Prioritize the highest-impact changes first.",
            "action": "Focus on TOP 3 priorities from recommendations above. Create accountability: (1) Share goals with friend/family, (2) Use habit-tracking app, (3) Schedule follow-up appointment in 3 months to reassess. Improvement in cycle regularity typically seen within 3-6 months of consistent lifestyle changes.",
            "priority": "HIGH"
        })
    elif wellness_score < 70:
        recommendations.append({
            "category": "⭐ Overall Wellness - Room for Improvement",
            "title": "Good Foundation - Fine-Tune Habits",
            "advice": f"Wellness score of {wellness_score:.0f} shows you're doing many things right! Focus on optimizing remaining areas.",
            "action": "Identify 1-2 areas for improvement from recommendations. Implement gradually over next 2-3 months. Track cycle to see if changes improve regularity.",
            "priority": "MODERATE"
        })
    elif wellness_score < 80:
        recommendations.append({
            "category": "⭐ Overall Wellness - Very Good",
            "title": "Strong Health Habits!",
            "advice": f"Excellent wellness score of {wellness_score:.0f}! You're taking great care of your health.",
            "action": "Continue current habits. Address any remaining minor concerns from recommendations. Consider becoming a health advocate to help others!",
            "priority": "LOW"
        })
    elif wellness_score >= 80:
        recommendations.append({
            "category": "⭐ Overall Wellness - Excellent!",
            "title": "Outstanding Health Status! 🌟",
            "advice": f"Exceptional wellness score of {wellness_score:.0f}! You're a role model for healthy living and cycle management.",
            "action": "Keep up the phenomenal work! Continue current lifestyle. Share your health journey to inspire others. Consider fine-tuning based on any remaining minor recommendations. You're thriving!",
            "priority": "LOW"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 🏥 PREVENTIVE CARE BY AGE
    # ═══════════════════════════════════════════════════════════════
    if age >= 21 and age < 30:
        recommendations.append({
            "category": "🏥 Preventive Care - Age 21-29",
            "title": "Regular Screening for Young Adults",
            "advice": "Women 21-29 need regular gynecological care even if healthy. Early detection prevents serious issues.",
            "action": "SCREENINGS: (1) Pap smear every 3 years (ages 21-29) - screens for cervical cancer/HPV, (2) STI testing if sexually active, (3) Clinical breast exam annually, (4) HPV vaccine if not completed (up to age 26, some coverage to 45), (5) Annual wellness visit with primary care, (6) Blood pressure check, (7) Cholesterol if risk factors. TRACK: Use period tracking app consistently. Report any cycle changes lasting >3 months.",
            "priority": "MODERATE"
        })
    elif age >= 30 and age < 40:
        recommendations.append({
            "category": "🏥 Preventive Care - Age 30-39",
            "title": "Comprehensive Screening for 30s",
            "advice": "Your 30s are crucial for reproductive health optimization and disease prevention.",
            "action": "SCREENINGS: (1) Pap smear + HPV co-testing every 5 years (or Pap alone every 3 years), (2) Annual wellness exam, (3) Mammogram if high risk (family history, BRCA genes), (4) Fertility assessment if planning pregnancy (AMH, FSH, antral follicle count) - especially if age 35+, (5) Thyroid screening (TSH), (6) Lipid panel, (7) Diabetes screening (fasting glucose, HbA1c) - especially if overweight/PCOS, (8) Vitamin D level, (9) Bone density if risk factors. FERTILITY: Fertility declines after 35. If trying to conceive >6 months without success at age 35+, see reproductive endocrinologist immediately. Consider egg freezing if delaying pregnancy.",
            "priority": "MODERATE"
        })
    elif age >= 40:
        recommendations.append({
            "category": "🏥 Preventive Care - Age 40+",
            "title": "Perimenopause Awareness & Increased Screening",
            "advice": "Women 40+ enter perimenopause transition (average age 47, but can start early 40s). Cycle irregularity increases. Cancer screening becomes critical.",
            "action": "SCREENINGS: (1) Mammogram ANNUALLY starting age 40 (screening mammography - earlier if high risk), (2) Pap smear + HPV co-testing every 5 years (can stop at 65 if prior normal screens), (3) Colonoscopy starting age 45 (colorectal cancer screening), (4) Annual wellness exam, (5) Lipid panel every 5 years, (6) Diabetes screening, (7) Bone density scan (DEXA) if risk factors or at menopause, (8) Thyroid screening. PERIMENOPAUSE: Expect cycle changes - skipped periods, shorter/longer cycles, heavier bleeding, new symptoms (hot flashes, night sweats, mood changes, sleep disruption). FSH >25 on day 3 suggests perimenopause. Consider hormone therapy if symptoms severe. Irregular bleeding at this age requires evaluation (endometrial biopsy) to rule out hyperplasia/cancer. Pregnancy still possible until menopause confirmed (12 months no period) - use contraception if needed.",
            "priority": "HIGH"
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 🏥 MEDICAL FOLLOW-UP
    # ═══════════════════════════════════════════════════════════════
    if pred_days >= 45 or cramp_severity >= 7 or has_pcos or has_endometriosis:
        recommendations.append({
            "category": "🏥 Medical Follow-Up",
            "title": "Track Progress & Re-evaluate",
            "advice": "Given your current health status, regular medical follow-up is essential to monitor progress and adjust treatment.",
            "action": "FOLLOW-UP SCHEDULE: (1) Return visit in 3 months to reassess cycle regularity and symptoms, (2) Repeat relevant blood work (hormones, thyroid, metabolic panel) in 3-6 months, (3) Track cycles meticulously using app (Clue, Flo, Fertility Friend), (4) Keep symptom diary, (5) Bring this report and tracking data to appointments, (6) If no improvement after 3-6 months of lifestyle changes, may need medication adjustments or specialist referral (reproductive endocrinologist), (7) Join support groups (PCOS, endometriosis) for community support and latest treatment info. Don't give up - cycle irregularities are treatable!",
            "priority": "MODERATE"
        })
    
    # Sort by priority: CRITICAL → HIGH → MODERATE → LOW
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "LOW"), 3))
    
    return recommendations
