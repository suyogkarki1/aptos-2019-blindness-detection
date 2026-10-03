"""Educational text shown in the UI, based on the International Clinical
Diabetic Retinopathy (ICDR) severity scale used by APTOS 2019."""

GRADE_INFO = {
    0: {
        "name": "No DR",
        "short": "No visible signs of diabetic retinopathy.",
        "color": "#2e7d32",
        "meaning": (
            "The retina looks healthy. Blood vessels are intact and there are no "
            "leaks, bleeds or abnormal vessels visible in the photograph."
        ),
        "signs": ["No microaneurysms, haemorrhages or exudates"],
        "action": "Routine screening, typically once a year.",
        "urgency": "Routine",
    },
    1: {
        "name": "Mild NPDR",
        "short": "Earliest stage: tiny bulges in small blood vessels.",
        "color": "#9e9d24",
        "meaning": (
            "Mild non-proliferative diabetic retinopathy. High blood sugar has "
            "started to weaken the walls of the smallest retinal vessels, which "
            "bulge out into tiny balloons called microaneurysms. Vision is "
            "usually unaffected."
        ),
        "signs": ["Microaneurysms only (tiny red dots)"],
        "action": "Re-examination in about 6–12 months and tighter control of blood sugar and blood pressure.",
        "urgency": "Monitor",
    },
    2: {
        "name": "Moderate NPDR",
        "short": "Vessels leak blood and fluid into the retina.",
        "color": "#ef6c00",
        "meaning": (
            "Moderate non-proliferative diabetic retinopathy. More vessels are "
            "damaged: they leak blood, fluid and fats into the retina, and some "
            "start to close off, starving areas of the retina of oxygen. Swelling "
            "of the central retina (macular oedema) can begin at this stage."
        ),
        "signs": [
            "Dot and blot haemorrhages",
            "Hard exudates (yellow fatty deposits)",
            "Cotton wool spots (fluffy white patches)",
        ],
        "action": "Referral to an eye specialist, usually within 3–6 months.",
        "urgency": "Refer",
    },
    3: {
        "name": "Severe NPDR",
        "short": "Large areas of retina are starved of blood supply.",
        "color": "#d84315",
        "meaning": (
            "Severe non-proliferative diabetic retinopathy. Many vessels are "
            "blocked and large parts of the retina are deprived of oxygen. The "
            "retina responds by releasing growth signals (VEGF), so there is a "
            "high risk of progressing to the proliferative stage within a year."
        ),
        "signs": [
            "More than 20 haemorrhages in each of the 4 quadrants",
            "Venous beading (vessels look like strings of sausages) in 2+ quadrants",
            "Intraretinal microvascular abnormalities (IRMA) in 1+ quadrant",
        ],
        "action": "Urgent referral to an eye specialist, usually within weeks.",
        "urgency": "Urgent",
    },
    4: {
        "name": "Proliferative DR",
        "short": "Fragile new vessels grow and can bleed or detach the retina.",
        "color": "#b71c1c",
        "meaning": (
            "Proliferative diabetic retinopathy, the most advanced stage. In "
            "response to oxygen starvation, new abnormal blood vessels grow on "
            "the retina and optic disc. They are fragile, can bleed into the "
            "vitreous gel and form scar tissue that pulls the retina off "
            "(retinal detachment), leading to severe vision loss."
        ),
        "signs": [
            "Neovascularisation (new vessels) on the disc or elsewhere",
            "Vitreous or pre-retinal haemorrhage",
            "Fibrous scar tissue",
        ],
        "action": (
            "Urgent referral. Treatments include anti-VEGF injections, laser "
            "photocoagulation and vitrectomy surgery."
        ),
        "urgency": "Urgent",
    },
}

CAUSES = [
    ("High blood sugar over time",
     "The main cause. Excess glucose damages the walls of the tiny vessels that feed the retina."),
    ("Duration of diabetes",
     "The longer someone has diabetes, the higher the risk. Most people with type 1 diabetes for 20+ years show some signs."),
    ("High blood pressure",
     "Adds extra strain on already weakened retinal vessels and speeds up damage."),
    ("High cholesterol",
     "Linked to more hard exudates (fatty deposits) and macular swelling."),
    ("Pregnancy",
     "Can make existing retinopathy progress quickly, so pregnant women with diabetes need closer eye checks."),
    ("Smoking and kidney disease",
     "Both damage small blood vessels throughout the body, including the eye."),
]

PROGRESSION = [
    ("1. Vessel walls weaken", "High glucose damages the cells lining small retinal vessels."),
    ("2. Leaks and bleeds", "Weak vessels bulge (microaneurysms) and leak blood, fluid and fats."),
    ("3. Vessels close off", "Blocked vessels leave parts of the retina without oxygen."),
    ("4. New vessels grow", "The starved retina releases VEGF, triggering fragile new vessels."),
    ("5. Bleeding and scarring", "New vessels bleed and form scar tissue that can detach the retina."),
]

SYMPTOMS = [
    "Often **no symptoms at all** in the early stages, which is why regular screening matters",
    "Blurred or fluctuating vision",
    "Floaters: dark spots or strings drifting across vision",
    "Dark or empty areas in the field of vision",
    "Difficulty seeing colours or at night",
    "Sudden vision loss (bleeding or retinal detachment)",
]

PREVENTION = [
    "Keep blood sugar in the target range (HbA1c as advised by a doctor)",
    "Control blood pressure and cholesterol",
    "Have a dilated eye exam or retinal photograph at least once a year",
    "Stop smoking",
    "See an eye doctor promptly if vision changes",
]
