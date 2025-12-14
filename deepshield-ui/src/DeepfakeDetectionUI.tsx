import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence, LayoutGroup } from "framer-motion";
import {
    Search,
    Camera,
    Upload,
    Moon,
    Sun,
    Languages,
    ShieldAlert,
    Activity,
    Mail,
    CheckCircle2,
    Cpu,
    BadgeCheck,
    Database,
    GraduationCap,
    Award,
    X,
} from "lucide-react";

/* ============================================================================
   DeepGuard — Image Deepfake Detection UI
   - Galaxy background with stars + comets
   - Light/Dark toggle
   - EN/AR copy
   - Auto analyze on file select (no separate Analyze button)
   - Consistent ids: dg-*
============================================================================ */

const copy = {
    en: {
        brand: "DeepGuard",
        tagline: "Detect deepfakes in images. Instantly.",
        subTag: "Upload an image and get your result !",
        ctaSecondary: "Upload image",
        nav: { product: "Product", ack: "Acknowledgements", contact: "Contact" },
        drop: {
            title: "Drag & drop your image",
            or: "or",
            browse: "browse files",
            note: "JPG, PNG — up to 10MB",
        },
        analyzing: "Analyzing…",
        result: {
            title: "Detection results",
            authenticity: "Authenticity",
            confidence: "Confidence",
            labelRealHigh: "Real",
            labelRealLow: "Likely Real",
            labelFakeHigh: "AI Generated",
            labelFakeLow: "Likely AI Generated",
            heatmap: "Explainability heatmap",
            preview: "Preview",
        },
        how: {
            title: "How it works",
            s1: "Upload",
            s1d: "Pixels are decoded securely on-device or in your private cloud.",
            s2: "Analyze",
            s2d: "Models scan spatial artifacts indicative of manipulations.",
            s3: "Explain",
            s3d: "Saliency and region-level heatmaps show why the score was assigned.",
            s4: "Decide",
            s4d: "Use thresholds to auto-flag risky media.",
        },
        ack: {
            title: "Acknowledgements",
            thanks:
                "This project was completed as part of our graduation. We thank our supervisors and the open-source community.",
            labels: { advisors: "Advisors", teammates: "Teammates", datasets: "Datasets", libs: "Libraries" },
            advisors: ["Dr. Ameur Touir"],
            teammates: ["Idriss Barbereau", "Eyad alqabbaa", "Abdullah AlEnezi", "Sleman Alissa"],
            datasets: ["Deepfake and real images dataset from Kaggle by Manjil Karki"],
            libs: ['Flask', 'TensorFlow', 'OpenCV', 'Numpy', 'transformers'],
        },
        footer: {
            rights: "All rights reserved.",
            privacy: "Privacy",
            terms: "Terms",
            security: "Security",
            madeIn: "Made in King Saud University (KSU)."
        },
        errors: { analyze: "An error occurred while analyzing the image." },
    },

    ar: {
        brand: "ديب غارد",
        tagline: "اكتشف التزييف العميق في الصور فورًا.",
        subTag: "ارفع صورة واحصل على نتيجة موثوقة مع خريطة حرارية تشرح سبب الحكم.",
        ctaSecondary: "رفع صورة",
        nav: { product: "المنتج", ack: "الشكر والتقدير", contact: "تواصل" },
        drop: {
            title: "اسحب وأضف الصورة هنا",
            or: "أو",
            browse: "تصفّح الملفات",
            note: "JPG, PNG — حتى 10 ميجابايت",
        },
        analyzing: "جارٍ التحليل...",
        result: {
            title: "نتيجة الفحص",
            authenticity: "المصداقية",
            confidence: "درجة الثقة",
            labelRealHigh: "حقيقية",
            labelRealLow: "غالبًا حقيقية",
            labelFakeHigh: "غير حقيقية",
            labelFakeLow: "غالبًا غير حقيقية",
            heatmap: "الخريطة الحرارية التوضيحية",
            preview: "معاينة",
        },
        how: {
            title: "آلية العمل",
            s1: "رفع الصورة",
            s1d: "تُفك رموز البكسلات بأمان محليًا أو في سحابتك الخاصة.",
            s2: "التحليل",
            s2d: "نرصد الآثار المكانية الدالة على التلاعب.",
            s3: "الشرح",
            s3d: "خرائط حرارية تبرز المناطق المشكوك فيها.",
            s4: "القرار",
            s4d: "حدود وسياسات لتمييز الوسائط الخطرة وتصدير تقارير موقعة.",
        },
        ack: {
            title: "الشكر والتقدير",
            thanks: "تم إنجاز هذا المشروع كمشروع تخرج. نشكر المشرفين ومجتمع المصادر المفتوحة.",
            labels: { advisors: "المشرفون", teammates: "أعضاء الفريق", datasets: "مجموعات البيانات", libs: "المكتبات" },
            advisors: ["د. عامر الطوير"],
            teammates: ["إدريس باربيرو", "إياد القباع", "عبدالله العنزي", "سليمان العيسى"],
            datasets: ["Deepfake and real images dataset from Kaggle by Manjil Karki"],
            libs: ['Flask', 'TensorFlow', 'OpenCV', 'Numpy', 'transformers'],
        },

        footer: {
            rights: "جميع الحقوق محفوظة.",
            privacy: "الخصوصية",
            terms: "الشروط",
            security: "الأمان",
            madeIn: "تم التطوير في جامعة الملك سعود (KSU)."
        },
        errors: { analyze: "حدث خطأ أثناء تحليل الصورة." },
    },
} as const;

const prettyPct = (n: number) => `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%`;

function useFakeInference(analyzing: boolean) {
    const [progress, setProgress] = useState(0);
    const [score, setScore] = useState(0.12);

    useEffect(() => {
        if (!analyzing) return;
        setProgress(0);
        const total = 60;
        let cur = 0;
        const id = setInterval(() => {
            cur += 2 + Math.round(Math.random() * 2);
            const p = Math.min(100, Math.round((cur / total) * 100));
            setProgress(p);
            if (p >= 100) {
                clearInterval(id);
                setScore(Math.random() * 0.6 + 0.2);
            }
        }, 80);
        return () => clearInterval(id);
    }, [analyzing]);

    return { progress, score, setScore };
}

type FeatureProps = {
    icon: any;
    title: string;
    desc: string;
    cardClass: string;
    subtleClass: string;
};
const Feature = ({ icon: Icon, title, desc, cardClass, subtleClass }: FeatureProps) => (
    <div className={`dg-card rounded-2xl border p-6 backdrop-blur ${cardClass}`}>
        <div className="flex items-center gap-3">
            <div className={`rounded-xl p-2 border ${subtleClass}`}>
                <Icon className="h-5 w-5" />
            </div>
            <h4 className="font-semibold text-lg">{title}</h4>
        </div>
        <p className="mt-3 text-sm opacity-80 leading-relaxed">{desc}</p>
    </div>
);

export default function App() {
    // ---------------- UI state
    const [dark, setDark] = useState(true);
    const [lang, setLang] = useState<"en" | "ar">("en");
    const t = useMemo(() => copy[lang], [lang]);
    const dir = lang === "ar" ? "rtl" : "ltr";

    const cardClass = dark ? "border-white/10 bg-white/5" : "border-slate-900/10 bg-white/70";
    const subtleClass = dark ? "border-white/15" : "border-slate-900/15";
    const pillClass = dark ? "bg-white/10 hover:bg-white/20" : "bg-slate-900/5 hover:bg-slate-900/10";

    // ---------------- File + results state
    const [file, setFile] = useState<File | null>(null);
    const [previewURL, setPreviewURL] = useState<string | null>(null);

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const { progress, score, setScore } = useFakeInference(isAnalyzing);

    const [serverPred, setServerPred] = useState<"fake" | "real" | null>(null);
    const [serverConf, setServerConf] = useState<number | null>(null);
    const [serverHeatmap, setServerHeatmap] = useState<string | null>(null);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    // ---------------- Helpers
    const isFake = score > 0.55;

    const conf = useMemo(() => (isFake ? score : 1 - score), [isFake, score]);
    const displayConf = serverConf !== null ? serverConf : conf;

    // --- LOGIC FOR LABELING BASED ON CONFIDENCE (Threshold: 0.75) ---
    const label = useMemo(() => {
        const isHighConfidence = displayConf >= 0.75;
        if (isFake) {
            return isHighConfidence ? t.result.labelFakeHigh : t.result.labelFakeLow;
        } else {
            return isHighConfidence ? t.result.labelRealHigh : t.result.labelRealLow;
        }
    }, [isFake, displayConf, t]);

    const onChoose = (f?: File) => {
        if (!f) return;
        setFile(f);
        setServerPred(null);
        setServerConf(null);
        setServerHeatmap(null);
        setErrorMsg(null);
        if (previewURL) URL.revokeObjectURL(previewURL);
        setPreviewURL(URL.createObjectURL(f));
    };

    const resetState = useCallback(() => {
        setFile(null);
        if (previewURL) URL.revokeObjectURL(previewURL);
        setPreviewURL(null);
        setIsAnalyzing(false);
        setServerPred(null);
        setServerConf(null);
        setServerHeatmap(null);
        setErrorMsg(null);
    }, [previewURL]);

    const analyzeImage = useCallback(async () => {
        if (!file) return;
        setIsAnalyzing(true);
        setErrorMsg(null);
        try {
            const formData = new FormData();
            formData.append("image", file);

            const resp = await fetch("https://sleman-one-deepguard-backend.hf.space/analyze", { method: "POST", body: formData });
            const data = await resp.json();

            const pred = (data?.prediction || "fake").toString().toLowerCase() as "fake" | "real";
            const c = typeof data?.confidence === "number" ? Math.max(0, Math.min(1, data.confidence)) : 0.85;

            setServerPred(pred);
            setServerConf(c);

            const pFake = pred === "fake" ? c : 1 - c;
            setScore(pFake);

            const hm = data?.heatmap || data?.heatmap_url || null;
            if (typeof hm === "string") setServerHeatmap(hm);
        } catch (e) {
            console.error(e);
        } finally {
            setIsAnalyzing(false);
        }
    }, [file, setScore, t.errors.analyze]);

    useEffect(() => {
        if (file && file.size > 0) analyzeImage();
    }, [file, analyzeImage]);

    useEffect(() => {
        return () => {
            if (previewURL) URL.revokeObjectURL(previewURL);
        };
    }, [previewURL]);

    useEffect(() => {
        document.documentElement.setAttribute("lang", lang);
        document.documentElement.setAttribute("dir", dir);
        document.title = lang === "ar" ? "ديب غارد" : "DeepGuard";
    }, [lang, dir]);

    const dropRef = useRef<HTMLDivElement | null>(null);
    useEffect(() => {
        const el = dropRef.current;
        if (!el) return;
        const prevent = (e: Event) => { e.preventDefault(); e.stopPropagation(); };
        const onDrop = (e: DragEvent) => {
            prevent(e);
            const f = e.dataTransfer?.files?.[0];
            if (f) onChoose(f);
        };
        ["dragenter", "dragover", "dragleave", "drop"].forEach((n) => el.addEventListener(n, prevent as any));
        el.addEventListener("drop", onDrop as any);
        return () => {
            ["dragenter", "dragover", "dragleave", "drop"].forEach((n) => el.removeEventListener(n, prevent as any));
            el.removeEventListener("drop", onDrop as any);
        };
    }, []);

    const stars = useMemo(
        () =>
            Array.from({ length: 160 }).map((_, i) => ({
                id: i,
                x: Math.random() * 100,
                y: Math.random() * 100,
                size: Math.random() < 0.8 ? 1 : 2,
                delay: Math.random() * 4,
                dur: 2 + Math.random() * 3,
            })),
        []
    );

    const comets = useMemo(
        () =>
            Array.from({ length: 8 }).map((_, i) => ({
                id: i,
                top: -10 - Math.random() * 25,
                left: 105 + Math.random() * 20,
                tx: -130 - Math.random() * 40,
                ty: 130 + Math.random() * 30,
                delay: Math.random() * 6,
                dur: 4 + Math.random() * 4,
                width: 120 + Math.random() * 120,
            })),
        []
    );

    const starColor = dark ? "rgba(255,255,255,0.95)" : "rgba(90,110,170,0.9)";
    const starGlow = dark ? "0 0 6px rgba(180,160,255,0.45)" : "0 0 6px rgba(120,150,255,0.35)";
    const cometTailDark =
        "linear-gradient(90deg, rgba(255,255,255,0.0) 0%, rgba(200,200,255,0.9) 60%, rgba(255,255,255,1) 100%)";
    const cometTailLight =
        "linear-gradient(90deg, rgba(255,255,255,0.0) 0%, rgba(140,170,255,0.9) 60%, rgba(200,220,255,1) 100%)";
    const cometHeadDark = "#ffffff";
    const cometHeadLight = "#9fb6ff";
    const cometGlowDark = "0 0 10px rgba(220,220,255,0.9)";
    const cometGlowLight = "0 0 10px rgba(150,170,255,0.9)";

    const hasFile = file !== null;

    return (
        <div id="dg-root">
            <div
                className={`relative min-h-screen selection:bg-violet-500/30 ${lang === "ar" ? "font-arabic" : "font-sans"} ${dark
                    ? "text-slate-100 bg-[linear-gradient(180deg,#0b1021,#0e1430_40%,#1a1635)]"
                    : "text-slate-800 bg-[linear-gradient(180deg,#ffffff,#eef6ff_40%,#fde7ff)]"
                    }`}
            >
                <div className="pointer-events-none fixed inset-0 -z-10">
                    <div className="absolute -top-40 left-1/2 h-[40rem] w-[40rem] -translate-x-1/2 rounded-full bg-gradient-to-br from-cyan-400/20 to-fuchsia-500/20 blur-[120px]" />
                </div>

                <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
                    {stars.map((s) => (
                        <span
                            key={s.id}
                            className="absolute rounded-full"
                            style={{
                                top: `${s.y}%`,
                                left: `${s.x}%`,
                                width: s.size,
                                height: s.size,
                                background: starColor,
                                boxShadow: starGlow,
                                borderRadius: 9999,
                                opacity: 0.8,
                                animation: `twinkle ${s.dur}s ease-in-out ${s.delay}s infinite`,
                            }}
                        />
                    ))}
                    {comets.map((c) => (
                        <div
                            key={c.id}
                            className="absolute"
                            style={
                                {
                                    top: `${c.top}%`,
                                    left: `${c.left}%`,
                                    transform: "rotate(-35deg)",
                                    filter: dark ? "drop-shadow(0 0 6px rgba(200,200,255,0.6))" : "drop-shadow(0 0 6px rgba(120,150,255,0.5))",
                                    animation: `shoot ${c.dur}s linear ${c.delay}s infinite`,
                                    "--tx": `${c.tx}vw`,
                                    "--ty": `${c.ty}vh`,
                                } as React.CSSProperties
                            }
                        >
                            <div className="tail" style={{ width: c.width, height: 2, borderRadius: 9999, background: dark ? cometTailDark : cometTailLight }} />
                            <div
                                className="head"
                                style={{
                                    position: "absolute",
                                    width: 4,
                                    height: 4,
                                    marginTop: -3,
                                    marginLeft: -2,
                                    background: dark ? cometHeadDark : cometHeadLight,
                                    borderRadius: 9999,
                                    boxShadow: dark ? cometGlowDark : cometGlowLight,
                                }}
                            />
                        </div>
                    ))}
                </div>

                <style>{`
          @keyframes twinkle { 0%,100%{opacity:.25} 50%{opacity:1} }
          @keyframes shoot {
            0% { transform: translate(0,0) rotate(-35deg); opacity:0; }
            10% { opacity:1; }
            90% { opacity:1; }
            100%{ transform: translate(var(--tx),var(--ty)) rotate(-35deg); opacity:0; }
          }
        `}</style>

                <div className="relative z-10">
                    <header id="dg-navbar" className="sticky top-0 z-30 backdrop-blur supports-[backdrop-filter]:bg-black/20">
                        <div className="mx-auto max-w-7xl px-4">
                            <div className="flex h-16 items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-violet-500/20 border border-violet-400/20">
                                        <Search className="h-5 w-5" />
                                    </div>
                                    <span className="font-semibold tracking-tight bg-gradient-to-r from-cyan-400 to-violet-500 bg-clip-text text-transparent">
                                        {t.brand}
                                    </span>
                                    <span className="hidden md:inline-block text-xs opacity-70 border rounded-full px-2 py-0.5 ms-2">beta</span>
                                </div>
                                <nav className="hidden md:flex items-center gap-6 text-sm opacity-90">
                                    <a href="#product" className="hover:opacity-100">{t.nav.product}</a>
                                    <a href="#ack" className="hover:opacity-100">{t.nav.ack}</a>
                                    <a href="#contact" className="hover:opacity-100">{t.nav.contact}</a>
                                </nav>
                                <div className="flex items-center gap-2">
                                    <button
                                        id="dg-lang-toggle"
                                        onClick={() => setLang((l) => (l === "en" ? "ar" : "en"))}
                                        className={`rounded-xl border px-3 py-2 flex items-center gap-2 ${pillClass} ${subtleClass}`}
                                        aria-label="Toggle language"
                                    >
                                        <Languages className="h-4 w-4" /> {lang.toUpperCase()}
                                    </button>
                                    <button
                                        id="dg-theme-toggle"
                                        onClick={() => setDark((d) => !d)}
                                        className={`rounded-xl border p-2 ${pillClass} ${subtleClass}`}
                                        aria-label="Toggle theme"
                                    >
                                        {dark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </header>

                    <section id="dg-hero" className="relative mx-auto max-w-7xl px-4 pt-8 md:pt-12">
                        <LayoutGroup>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8 items-center relative">
                                <AnimatePresence mode="popLayout">
                                    {!hasFile && (
                                        <motion.div
                                            key="hero-left-content"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: -50, scale: 0.95, filter: "blur(10px)" }}
                                            transition={{ duration: 0.4, ease: "easeInOut" }}
                                            style={{ gridColumn: 1 }}
                                        >
                                            <h1 className="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight leading-[1.05]">
                                                {t.tagline}
                                            </h1>
                                            <p className="mt-4 text-base md:text-lg opacity-80 max-w-prose">{t.subTag}</p>

                                            <div className="mt-6 flex flex-col sm:flex-row gap-3">
                                                <label
                                                    htmlFor="dg-file-input"
                                                    className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-violet-500 px-5 py-3 font-semibold hover:from-cyan-400 hover:to-violet-600 transition-transform active:scale-95"
                                                >
                                                    <Upload className="h-5 w-5" /> {t.ctaSecondary}
                                                </label>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>

                                <input
                                    id="dg-file-input"
                                    type="file"
                                    className="hidden"
                                    accept="image/*"
                                    capture="environment"
                                    onChange={(e) => onChoose(e.target.files?.[0] ?? undefined)}
                                />

                                <motion.div
                                    id="product"
                                    layout
                                    transition={{ layout: { duration: 0.5, type: "spring", bounce: 0.15 } }}
                                    className={`dg-upload-card rounded-2xl border p-4 md:p-6 backdrop-blur ${cardClass} ${hasFile ? "md:col-span-2 w-full" : "md:col-span-1"}`}
                                    style={{
                                        zIndex: hasFile ? 20 : 1,
                                        gridColumnStart: hasFile ? 1 : "auto"
                                    }}
                                >
                                    <AnimatePresence initial={false} mode="wait">
                                        {!hasFile && (
                                            <motion.div
                                                key="dropzone-container"
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1, height: 'auto' }}
                                                exit={{ opacity: 0, height: 0, overflow: 'hidden', marginBottom: 0 }}
                                                transition={{ duration: 0.3 }}
                                            >
                                                <div
                                                    id="dg-dropzone"
                                                    ref={dropRef}
                                                    className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors ${subtleClass} hover:bg-white/5`}
                                                >
                                                    <div className={`mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border ${subtleClass}`}>
                                                        <Camera className="h-6 w-6" />
                                                    </div>
                                                    <h3 className="text-lg font-semibold">{t.drop.title}</h3>
                                                    <p className="mt-1 text-xs opacity-70">{t.drop.note}</p>
                                                    <div className="mt-4 flex items-center justify-center gap-2 text-sm opacity-80">
                                                        <span>{t.drop.or}</span>
                                                        <label
                                                            htmlFor="dg-file-input"
                                                            className={`cursor-pointer rounded-lg border px-2 py-1 transition-colors ${pillClass} ${subtleClass}`}
                                                        >
                                                            {t.drop.browse}
                                                        </label>
                                                    </div>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>

                                    <AnimatePresence mode="wait">
                                        {hasFile && (
                                            <motion.div
                                                key="results-container"
                                                initial={{ opacity: 0, y: 20 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0, y: -20 }}
                                                transition={{ duration: 0.4, delay: 0.1 }}
                                                className="w-full"
                                            >
                                                <div className="flex items-center justify-between mb-4">
                                                    <div className="flex items-center gap-2">
                                                        {(!isAnalyzing && isFake) ? (
                                                            <ShieldAlert className="h-5 w-5 text-rose-400" />
                                                        ) : (!isAnalyzing && !isFake) ? (
                                                            <BadgeCheck className="h-5 w-5 text-emerald-400" />
                                                        ) : (
                                                            <Activity className="h-5 w-5 text-violet-400" />
                                                        )}
                                                        <h4 className="font-semibold text-lg">{isAnalyzing ? t.analyzing : t.result.title}</h4>
                                                    </div>
                                                    <button
                                                        onClick={resetState}
                                                        className={`rounded-full p-1.5 border ${pillClass} ${subtleClass} transition-transform hover:rotate-90`}
                                                        aria-label="Close results"
                                                    >
                                                        <X className="h-4 w-4 opacity-70" />
                                                    </button>
                                                </div>

                                                <AnimatePresence>
                                                    {isAnalyzing && (
                                                        <motion.div
                                                            id="dg-progress"
                                                            initial={{ opacity: 0, height: 0 }}
                                                            animate={{ opacity: 1, height: 'auto' }}
                                                            exit={{ opacity: 0, height: 0 }}
                                                            className="mb-6"
                                                        >
                                                            <div className="flex items-center justify-between text-sm mb-2 opacity-80">
                                                                <span>{t.analyzing}</span>
                                                                <span>{progress}%</span>
                                                            </div>
                                                            <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                                                                <div className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all ease-out" style={{ width: `${progress}%` }} />
                                                            </div>
                                                        </motion.div>
                                                    )}
                                                </AnimatePresence>

                                                <AnimatePresence>
                                                    {!isAnalyzing && file && (
                                                        <motion.div
                                                            id="dg-results-grid"
                                                            initial={{ opacity: 0, scale: 0.98 }}
                                                            animate={{ opacity: 1, scale: 1 }}
                                                            transition={{ duration: 0.4 }}
                                                            className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6"
                                                        >
                                                            <div id="dg-auth-card" className={`rounded-xl border p-4 ${cardClass} flex flex-col justify-between`}>
                                                                <div>
                                                                    <div className="flex justify-between items-start">
                                                                        <div className="text-sm opacity-70">{t.result.authenticity}</div>
                                                                        <div className="text-xs opacity-50 truncate max-w-[100px]">{file?.name}</div>
                                                                    </div>
                                                                    <div className={`mt-2 text-4xl font-bold tracking-tight ${isFake ? "text-rose-400" : "text-emerald-400"}`}>{label}</div>
                                                                </div>

                                                                <div className="mt-6">
                                                                    <div className="flex justify-between text-xs font-semibold uppercase tracking-wider mb-2 opacity-80">
                                                                        <span>Confidence Level</span>
                                                                        <span className={isFake ? "text-rose-400" : "text-emerald-400"}>
                                                                            {prettyPct(displayConf)}
                                                                        </span>
                                                                    </div>

                                                                    <div className="h-6 w-full overflow-hidden rounded-full bg-slate-900/40 border border-white/5 relative">
                                                                        <motion.div
                                                                            className={`h-full absolute left-0 top-0 ${isFake ? "bg-gradient-to-r from-rose-600 to-rose-400" : "bg-gradient-to-r from-emerald-600 to-emerald-400"}`}
                                                                            initial={{ width: 0 }}
                                                                            animate={{ width: `${displayConf * 100}%` }}
                                                                            transition={{ duration: 1, ease: "easeOut" }}
                                                                        />
                                                                        <div className="absolute inset-0 opacity-20 bg-[linear-gradient(45deg,rgba(255,255,255,0.15)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.15)_50%,rgba(255,255,255,0.15)_75%,transparent_75%,transparent)] bg-[length:1rem_1rem]" />
                                                                    </div>

                                                                    <div className="flex justify-between mt-2 text-[10px] opacity-50 uppercase font-medium">
                                                                        <span>0%</span>
                                                                        <span>50%</span>
                                                                        <span>100%</span>
                                                                    </div>
                                                                </div>

                                                                {errorMsg && <div className="mt-3 text-sm text-rose-400 bg-rose-950/30 p-2 rounded border border-rose-900/50">{errorMsg}</div>}
                                                            </div>

                                                            <div id="dg-preview-card" className={`col-span-1 md:col-span-1 rounded-xl border p-1 ${cardClass}`}>
                                                                {/* CONTRAST FIX: Added bg-black to ensure visibility in light mode */}
                                                                <div className="relative h-64 md:h-80 w-full rounded-lg overflow-hidden bg-black">
                                                                    {previewURL ? (
                                                                        <img
                                                                            id="dg-preview-image"
                                                                            src={previewURL}
                                                                            alt="preview"
                                                                            className="absolute inset-0 h-full w-full object-cover"
                                                                        />
                                                                    ) : (
                                                                        <div className="absolute inset-0 grid place-items-center text-sm opacity-60">
                                                                            {lang === "ar" ? "لا توجد معاينة" : "No preview"}
                                                                        </div>
                                                                    )}
                                                                    <div className="absolute top-2 left-2 rounded-md bg-black/40 backdrop-blur-md px-2 py-1 text-[10px] border border-white/10">
                                                                        {t.result.preview}
                                                                    </div>
                                                                </div>
                                                            </div>

                                                            <div id="dg-heatmap-card" className={`col-span-1 md:col-span-1 rounded-xl border p-1 ${cardClass}`}>
                                                                {/* CONTRAST FIX: Added bg-black to ensure visibility in light mode */}
                                                                <div className="relative h-64 md:h-80 w-full rounded-lg overflow-hidden bg-black">
                                                                    {serverHeatmap ? (
                                                                        <>
                                                                            {previewURL && (
                                                                                <img
                                                                                    src={previewURL}
                                                                                    alt="preview underlay"
                                                                                    className="absolute inset-0 h-full w-full object-cover opacity-50 blur-sm"
                                                                                />
                                                                            )}
                                                                            <img
                                                                                src={serverHeatmap}
                                                                                alt="heatmap"
                                                                                className="absolute inset-0 h-full w-full object-contain mix-blend-screen"
                                                                            />
                                                                        </>
                                                                    ) : (
                                                                        <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/10 via-violet-500/10 to-fuchsia-500/10 animate-pulse">
                                                                            <div className="absolute inset-0 grid grid-cols-4 grid-rows-3 opacity-30">
                                                                                {Array.from({ length: 12 }).map((_, i) => (
                                                                                    <motion.div
                                                                                        key={i}
                                                                                        className="rounded-full blur-xl"
                                                                                        animate={{
                                                                                            backgroundColor: ['rgba(255,0,120,0.1)', 'rgba(0,255,255,0.1)', 'rgba(255,0,120,0.1)'],
                                                                                            scale: [1, 1.2, 1]
                                                                                        }}
                                                                                        transition={{ duration: 3 + Math.random() * 2, repeat: Infinity, delay: Math.random() * 2 }}
                                                                                        style={{ background: `rgba(255,0,120,${Math.random() * 0.2})` }}
                                                                                    />
                                                                                ))}
                                                                            </div>
                                                                        </div>
                                                                    )}
                                                                    <div className="absolute top-2 left-2 rounded-md bg-black/40 backdrop-blur-md px-2 py-1 text-[10px] border border-white/10">
                                                                        {t.result.heatmap}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </motion.div>
                                                    )}
                                                </AnimatePresence>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </motion.div>
                            </div>
                        </LayoutGroup>
                    </section>

                    <section className="mx-auto max-w-7xl px-4 mt-8 relative z-10">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <Feature icon={Upload} title={t.how.s1} desc={t.how.s1d} cardClass={cardClass} subtleClass={subtleClass} />
                            <Feature icon={Cpu} title={t.how.s2} desc={t.how.s2d} cardClass={cardClass} subtleClass={subtleClass} />
                            <Feature icon={BadgeCheck} title={t.how.s3} desc={t.how.s3d} cardClass={cardClass} subtleClass={subtleClass} />
                            <Feature icon={CheckCircle2} title={t.how.s4} desc={t.how.s4d} cardClass={cardClass} subtleClass={subtleClass} />
                        </div>
                    </section>

                    <section id="ack" className="mx-auto max-w-7xl px-4 mt-16 relative z-10">
                        <div className={`rounded-2xl border p-6 md:p-8 ${cardClass}`}>
                            <h3 className="text-2xl font-bold flex items-center gap-2">
                                <GraduationCap className="h-5 w-5" /> {t.ack.title}
                            </h3>
                            <p className="mt-2 opacity-80 text-sm md:text-base max-w-3xl">{t.ack.thanks}</p>

                            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className={`rounded-2xl border p-4 ${cardClass}`}>
                                    <div className="flex items-center gap-2 font-semibold mb-2">
                                        <Award className="h-4 w-4" /> {t.ack.labels.advisors}
                                    </div>
                                    <ul className="list-disc ms-5 space-y-1 text-sm opacity-90">
                                        {t.ack.advisors.map((n, i) => <li key={i}>{n}</li>)}
                                    </ul>
                                </div>

                                <div className={`rounded-2xl border p-4 ${cardClass}`}>
                                    <div className="flex items-center gap-2 font-semibold mb-2">
                                        <GraduationCap className="h-4 w-4" /> {t.ack.labels.teammates}
                                    </div>
                                    <ul className="list-disc ms-5 space-y-1 text-sm opacity-90">
                                        {t.ack.teammates.map((n, i) => <li key={i}>{n}</li>)}
                                    </ul>
                                </div>

                                <div className={`rounded-2xl border p-4 ${cardClass}`}>
                                    <div className="flex items-center gap-2 font-semibold mb-2">
                                        <Database className="h-4 w-4" /> {t.ack.labels.datasets}
                                    </div>
                                    <ul className="list-disc ms-5 space-y-1 text-sm opacity-90">
                                        {t.ack.datasets.map((n, i) => <li key={i}>{n}</li>)}
                                    </ul>
                                </div>

                                <div className={`rounded-2xl border p-4 ${cardClass}`}>
                                    <div className="flex items-center gap-2 font-semibold mb-2">
                                        <Cpu className="h-4 w-4" /> {t.ack.labels.libs}
                                    </div>
                                    <ul className="list-disc ms-5 space-y-1 text-sm opacity-90">
                                        {t.ack.libs.map((n, i) => <li key={i}>{n}</li>)}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </section>

                    <footer id="contact" className="mx-auto max-w-7xl px-4 mt-20 md:mt-28 pb-12 relative z-10">
                        <div className={`rounded-2xl border p-6 md:p-8 ${cardClass}`}>
                            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-violet-500/20 border border-violet-400/20">
                                            <Search className="h-5 w-5" />
                                        </div>
                                        <span className="font-semibold tracking-tight bg-gradient-to-r from-cyan-400 to-violet-500 bg-clip-text text-transparent">
                                            {t.brand}
                                        </span>
                                    </div>
                                    <p className="mt-2 text-sm opacity-80 max-w-prose">{t.footer.madeIn}</p>
                                </div>
                                <div className="flex flex-col gap-3 w-full md:w-auto">
                                    <a
                                        className={`inline-flex items-center gap-2 rounded-xl border px-4 py-2 ${pillClass} ${subtleClass}`}
                                        href="mailto:hello@deepguard.ai"
                                    >
                                        <Mail className="h-4 w-4" /> hello@deepguard.ai
                                    </a>
                                </div>
                            </div>
                            <div className="mt-6 text-xs opacity-60">
                                © {new Date().getFullYear()} {t.brand}. {t.footer.rights}
                            </div>
                        </div>
                    </footer>
                </div>
            </div>
        </div>
    );
}