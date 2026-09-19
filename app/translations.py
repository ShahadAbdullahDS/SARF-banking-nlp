INTENT_LABELS_EN = {
    "آبل باي أو جوجل باي": "Apple Pay or Google Pay",
    "إرجاع الدفع بالبطاقة؟": "Card payment refund?",
    "إعادة الشحن عن طريق رسوم التحويل المصرفي": "Top-up by bank transfer charge",
    "إلغاء التحويل": "Cancel transfer",
    "ابتلاع البطاقة": "Card swallowed",
    "استرداد الأموال غير مرئي": "Refund not showing",
    "البطاقات والعملات المدعومة": "Supported cards and currencies",
    "البطاقة الافتراضية لا تعمل": "Virtual card not working",
    "البطاقة المفقودة أو المسروقة": "Lost or stolen card",
    "البطاقة لا تعمل": "Card not working",
    "التحقق من مصدر الأموال": "Verify source of funds",
    "التعبئة عن طريق البطاقة": "Top-up by card",
    "التعبئة عن طريق شحن البطاقة": "Top-up by card charge",
    "الحصول على بطاقة احتياطية": "Get a spare card",
    "الحصول على بطاقة افتراضية": "Get a virtual card",
    "الحصول على بطاقة افتراضية يمكن التخلص منها": "Get a disposable virtual card",
    "الحصول على بطاقة فعلية": "Get a physical card",
    "الدفع المباشر للدين غير معروف": "Direct debit not recognized",
    "الدفع بالبطاقة المعلق": "Pending card payment",
    "الدفع بالبطاقة غير معترف به": "Card payment not recognized",
    "السحب النقدي غير معترف به": "Cash withdrawal not recognized",
    "الصرف عبر التطبيق": "Exchange via app",
    "المستفيد غير مسموح به": "Beneficiary not allowed",
    "الهاتف المفقود أو المسروق": "Lost or stolen phone",
    "انتظار التحويل": "Pending transfer",
    "انهاء حساب": "Terminate account",
    "بطاقة على وشك الانتهاء": "Card about to expire",
    "بطاقة مخترقة": "Compromised card",
    "تحرير التفاصيل الشخصية": "Edit personal details",
    "تحصيل رسوم التحويل": "Transfer fee charged",
    "تحقق من تعبئة الرصيد": "Verify top-up",
    "تحقق من هويتي": "Verify my identity",
    "تعبئة الرصيد نقدًا أو بشيك": "Top-up by cash or cheque",
    "تعبئة تلقائية": "Automatic top-up",
    "تغيير رمز التعريف الشخصي": "Change PIN",
    "تفعيل بطاقتي": "Activate my card",
    "تقدير تسليم البطاقة": "Card delivery estimate",
    "تلقي الأموال": "Receiving money",
    "تلقي مبلغ خاطئ من النقد": "Wrong amount of cash received",
    "تم التحصيل مرتين": "Charged twice",
    "تم تحصيل رسوم الدفع بالبطاقة": "Card payment fee charged",
    "تم رفض الدفع بالبطاقة": "Declined card payment",
    "توقيت التحويل": "Transfer timing",
    "حد السن": "Age limit",
    "حدود البطاقة التي يمكن التخلص منها": "Disposable card limits",
    "حدود التعبئة": "Top-up limits",
    "دعم أجهزة الصراف الآلي": "ATM support",
    "دعم البلد": "Country support",
    "دعم العملات الورقية": "Fiat currency support",
    "ربط البطاقة": "Link card",
    "رسوم إضافية على كشف الحساب": "Extra charge on statement",
    "رسوم السحب النقدي": "Cash withdrawal fee",
    "رسوم الصرف": "Exchange fee",
    "رفض التحويل": "Declined transfer",
    "رفض السحب النقدي": "Declined cash withdrawal",
    "رمز التعريف الشخصي محظور": "PIN blocked",
    "سعر الصرف": "Exchange rate",
    "سعر الصرف الخاطئ للدفع بالبطاقة": "Wrong exchange rate for card payment",
    "سعر صرف خاطئ للسحب النقدي": "Wrong exchange rate for cash withdrawal",
    "طلب استرداد": "Request refund",
    "طلب بطاقة فعلية": "Order physical card",
    "عادت تعبئة الرصيد": "Top-up reverted",
    "عدم التلامس لا يعمل": "Contactless not working",
    "غير قادر على التحقق من الهوية": "Unable to verify identity",
    "فشل التحويل": "Failed transfer",
    "فشل التعبئة": "Failed top-up",
    "في انتظار التعبئة": "Pending top-up",
    "في انتظار السحب النقدي": "Pending cash withdrawal",
    "فيزا أو ماستر كارد": "Visa or Mastercard",
    "قبول البطاقة": "Card acceptance",
    "لم يتم تحديث الرصيد بعد التحويل المصرفي": "Balance not updated after bank transfer",
    "لم يتم تحديث الرصيد بعد الشيك أو الإيداع النقدي": "Balance not updated after cheque or cash deposit",
    "لم يستلم المستلم التحويل": "Recipient not received transfer",
    "لماذا التحقق من الهوية": "Why verify identity",
    "نسيان رمز المرور": "Forgot passcode",
    "نقل إلى الحساب": "Transfer to account",
    "وصول البطاقة": "Card arrival",
}


def translate_label(label: str, lang: str) -> str:
    if lang == "en":
        return INTENT_LABELS_EN.get(label, label)
    return label


TEXTS = {
    "en": {
        "nav_try": "Try the Model",
        "nav_results": "Understand the Results",

        "app_title": "SARF",
        "app_subtitle": "Arabic Banking Intent Classifier",
        "badge": "Research prototype · AraBERTv2 · MSA to Saudi Arabic",
        "hero_text": "Type a banking sentence and watch how the model classifies it across 77 intents.",

        "input_placeholder": "Type your banking query here...",
        "try_sample": "Quick examples:",
        "sample_1": "My card isn't working",
        "sample_2": "I was charged twice",
        "sample_3": "I forgot my passcode",
        "sample_4": "My transfer didn't arrive",

        "kpi_intents": "Supported Intents",
        "kpi_f1": "Saudi Macro F1 · E1 Mean (3 Seeds)",
        "kpi_model": "Demo Checkpoint",
        "kpi_latency": "Warm Inference Benchmark",

        "sec_understood": "What did the model understand?",
        "model_understood": "The model believes this sentence is about:",
        "model_score": "Model score",

        "sec_decision": "Is the decision clear?",
        "decision_note_clear": "Clear relative ranking: the top intent leads by a wide margin.",
        "decision_note_close": "Close alternatives: the gap is small, and the second intent may be worth reviewing.",
        "top_intent_label": "Top intent",
        "second_intent_label": "Second intent",
        "gap_label": "Gap",
        "points_label": "points",

        "sec_alternatives": "What other intents did the model consider?",
        "col_rank": "#",
        "col_intent": "Intent",
        "col_score": "Model Score",

        "sec_retrieval": "Which MSA training examples are most similar?",
        "retrieval_note": (
            "These are the closest examples from the approved MSA training data. "
            "They are diagnostic only and do not change AraBERT's prediction."
        ),
        "retrieval_example": "MSA training example",
        "retrieval_intent": "Training intent",
        "retrieval_similarity": "Similarity",
        "retrieval_span_one": "The top 3 examples represent one training intent.",
        "retrieval_span_many": "The top 3 examples represent {count} training intents.",
        "retrieval_unavailable": "Similar training examples are currently unavailable.",

        "sec_confusion": "Historical confusion patterns",
        "confusion_intro": "In earlier evaluation, this intent was most often confused with:",
        "no_confusion": (
            "No confusion pair for this intent appears in the displayed "
            "high-frequency list."
        ),
        "times": "times",

        "sec_advanced": "Show advanced analysis",
        "advanced_context": "Evaluation context: Saudi held-out test · E1 seed 2026",
        "advanced_f1": "Intent Performance",
        "advanced_seeds": "Seed Stability (E1)",
        "response_time": "Response time",

        "reset_btn": "Clear",

        "results_title": "Understand the Results",
        "results_intro": (
            "Explore the aggregate performance, data characteristics, "
            "and model behavior across all 77 intents."
        ),

        "sec_model_compare": "How does SARF compare to other approaches?",
        "model_compare_finding": (
            "AraBERT achieved higher performance than traditional models "
            "on the final Saudi evaluation."
        ),

        "sec_f1_dist": "Was performance equal across all intents?",
        "f1_dist_finding": (
            "No. Some intents are easier to classify than others, "
            "so the overall average doesn't tell the whole story."
        ),

        "sec_intent_dist": "How was the training data distributed?",
        "intent_dist_finding": (
            "The chart shows the 20 most represented intents "
            "in the MSA training set."
        ),

        "sec_top_confusions": "Which intents does the model confuse most?",
        "confusions_finding": (
            "These are the most frequent true-to-predicted errors in the Saudi test. "
            "Similar wording or closely related banking topics may contribute "
            "to these mistakes."
        ),

        "sec_text_length": "Text length characteristics",
        "text_length_finding": (
            "How long are the texts in each split? The 95th percentile shows "
            "the length below which 95% of texts fall: only 5% are longer."
        ),

        "methodology_title": "Methodology",
        "meth_dataset": "**Dataset**: 10,732 train / 1,229 val / 3,580 Saudi test",
        "meth_model": "**Model** : AraBERTv2 (aubmindlab/bert-base-arabertv2)",
        "meth_checkpoint": (
            "**Checkpoint** : E1 / seed 2026, chosen by highest MSA validation F1 "
            "*before* seeing Saudi test"
        ),
        "meth_conditions": "**Conditions** : 5 (E0-E3 + EB) x 3 seeds (42, 123, 2026) = 15 experiments",
        "meth_transfer": "**Transfer** : MSA to Saudi cross-dialect evaluation",
        "meth_retrieval": (
            "**Diagnostic retrieval** : A multilingual sentence encoder compares each "
            "input with the approved MSA training examples. "
            "This does not affect AraBERT's decision."
        ),

        "limitations_title": "Limitations",
        "lim_calibration": "Scores are softmax outputs, not calibrated probabilities",
        "lim_oos": "No out-of-scope detection: unknown intents still get classified",
        "lim_routing": "No confidence threshold or routing policy",
        "lim_ensemble": "Single checkpoint, no ensemble",
        "lim_privacy": (
            "User inputs are processed in memory; this prototype does not include "
            "a database or file-based storage for user queries."
        ),

        "future_work_title": "Future Work",
        "future_work_1": (
            "Calibrate confidence scores and define an abstain-or-route policy for "
            "low-confidence predictions."
        ),
        "future_work_2": (
            "Add out-of-scope and unknown-intent detection so unsupported requests "
            "are not forced into one of the 77 supported intents."
        ),
        "future_work_3": (
            "Expand and audit Saudi banking examples, especially recurring confusion "
            "pairs, and evaluate any intent consolidation through a documented "
            "retraining protocol."
        ),
        "future_work_4": (
            "Compare parameter-efficient adaptation methods such as LoRA, larger "
            "carefully governed CPT corpora, and ensembles under the same held-out "
            "evaluation protocol."
        ),

        "footer": "SARF v1.0 · September 2026 · Inference-only prototype",

        "no_data": "No data available.",
        "f1_label": "F1",
        "precision_label": "Precision",
        "recall_label": "Recall",
        "support_label": "Samples",

        "tip_understood": (
            "The model ranks 77 banking intents and selects the highest-scoring one. "
            "The percentage is a softmax score: it shows the model's relative preference "
            "among the available intents, not a guaranteed probability of correctness."
        ),
        "tip_decision": (
            "This shows the difference between the first and second softmax scores. "
            "A large gap means the model ranked the first intent far above the second "
            "for this input; it does not guarantee that the prediction is correct."
        ),
        "tip_alternatives": (
            "The full ranking of all 77 intents by softmax score. Only the top 5 are shown. "
            "Scores across all intents sum to 100%."
        ),
        "tip_confusion": (
            "This section shows displayed high-frequency Saudi-test confusion pairs "
            "involving the predicted intent. If no pair appears, it means none was "
            "included in the saved top-confusion list; it does not prove the intent "
            "was never confused."
        ),
        "tip_retrieval": (
            "Cosine similarity between the input sentence and examples from the approved "
            "MSA training data, computed with a multilingual sentence encoder. "
            "A higher score means closer semantic wording. This is diagnostic only "
            "and does not determine AraBERT's prediction."
        ),
        "tip_model_compare": (
            "Each bar shows the mean Saudi Macro F1 across 3 random seeds (42, 123, 2026). "
            "AraBERT conditions (E0-E3, EB) differ in the amount of synthetic Saudi CPT data. "
            "TF-IDF+SVM and TextCNN are non-transformer baselines."
        ),
        "tip_f1_dist": (
            "Per-class F1 scores for the demo checkpoint (E1, seed 2026) on the Saudi test set. "
            "The histogram shows how many of the 77 intents fall in each F1 range. "
            "The dashed line marks the mean."
        ),
        "tip_confusions": (
            "The most frequent true-to-predicted misclassification pairs in the Saudi test set. "
            "High counts point to semantically close intent pairs that the model struggles "
            "to separate."
        ),
        "tip_intent_dist": (
            "Number of MSA training examples per intent. Imbalanced classes can lead to "
            "lower F1 on under-represented intents."
        ),
        "tip_text_length": (
            "Word-count statistics for each data split: MSA train, MSA val, MSA test, "
            "and Saudi test. The 95th percentile shows the length below which 95% of "
            "texts fall."
        ),
    },

    "ar": {
        "nav_try": "جرّب النموذج",
        "nav_results": "افهم النتائج",

        "app_title": "صَرْف",
        "app_subtitle": "مصنّف النوايا المصرفية العربية",
        "badge": "نموذج بحثي · AraBERTv2 · فصحى ← سعودي",
        "hero_text": "اكتب جملة مصرفية، وشاهد كيف يصنفها النموذج ضمن 77 نية.",

        "input_placeholder": "اكتب استفسارك البنكي هنا...",
        "try_sample": "أمثلة سريعة:",
        "sample_1": "بطاقتي ما تشتغل",
        "sample_2": "اتخصم مني مرتين",
        "sample_3": "نسيت الرقم السري",
        "sample_4": "حوّلتي ما وصلت",

        "kpi_intents": "نية مدعومة",
        "kpi_f1": "متوسط F1 الكلي · سعودي · E1 (3 بذور)",
        "kpi_model": "نموذج العرض",
        "kpi_latency": "قياس زمني بعد تحميل النموذج",

        "sec_understood": "ماذا فهم النموذج؟",
        "model_understood": "النموذج فهم أن الجملة تتعلق بـ:",
        "model_score": "درجة النموذج",

        "sec_decision": "هل القرار واضح؟",
        "decision_note_clear": "ترتيب واضح نسبيًا: النية الأولى تتقدم بفارق واسع.",
        "decision_note_close": "بدائل متقاربة: الفارق صغير، والنية الثانية قد تستحق المراجعة.",
        "top_intent_label": "النية الأولى",
        "second_intent_label": "النية الثانية",
        "gap_label": "الفارق",
        "points_label": "نقطة",

        "sec_alternatives": "ما النوايا الأخرى التي فكر فيها النموذج؟",
        "col_rank": "#",
        "col_intent": "النية",
        "col_score": "درجة النموذج",

        "sec_retrieval": "ما أمثلة التدريب بالفصحى الأقرب لهذه الجملة؟",
        "retrieval_note": (
            "هذه أقرب أمثلة من بيانات التدريب المعتمدة بالفصحى. "
            "تُعرض لأغراض الفهم والتحليل فقط، ولا تغيّر قرار AraBERT."
        ),
        "retrieval_example": "مثال من تدريب الفصحى",
        "retrieval_intent": "نية التدريب",
        "retrieval_similarity": "درجة التشابه",
        "retrieval_span_one": "تنتمي الأمثلة الثلاثة الأقرب إلى نية تدريب واحدة.",
        "retrieval_span_many": "تنتمي الأمثلة الثلاثة الأقرب إلى {count} نوايا تدريب مختلفة.",
        "retrieval_unavailable": "أمثلة التدريب المتقاربة غير متاحة حاليًا.",

        "sec_confusion": "أنماط التباس سابقة",
        "confusion_intro": "في التقييم السابق، كانت هذه النية تُخلط غالبًا مع:",
        "no_confusion": (
            "لا يظهر لهذه النية زوج التباس ضمن قائمة حالات الالتباس "
            "الأعلى تكرارًا المعروضة."
        ),
        "times": "مرة",

        "sec_advanced": "عرض التحليل المتقدم",
        "advanced_context": "سياق التقييم: اختبار سعودي محجوز · E1 seed 2026",
        "advanced_f1": "أداء النية",
        "advanced_seeds": "استقرار البذور (E1)",
        "response_time": "زمن الاستجابة",

        "reset_btn": "مسح",

        "results_title": "افهم النتائج",
        "results_intro": (
            "استكشف الأداء الإجمالي وخصائص البيانات وسلوك النموذج "
            "عبر كل 77 نية."
        ),

        "sec_model_compare": "كيف يقارن SARF بالنماذج الأخرى؟",
        "model_compare_finding": (
            "AraBERT حقق أداءً أعلى من النماذج التقليدية "
            "على التقييم السعودي النهائي."
        ),

        "sec_f1_dist": "هل كان الأداء متساويًا على كل النوايا؟",
        "f1_dist_finding": "لا. بعض النوايا أسهل من غيرها، لذلك المتوسط وحده لا يكفي.",

        "sec_intent_dist": "كيف توزعت بيانات التدريب؟",
        "intent_dist_finding": "يعرض الرسم أكثر 20 نية تمثيلًا في بيانات التدريب بالفصحى.",

        "sec_top_confusions": "ما أكثر النوايا التي يخلط النموذج بينها؟",
        "confusions_finding": (
            "تعرض هذه القائمة أكثر أخطاء التصنيف تكرارًا في الاختبار السعودي. "
            "وقد يساهم تقارب الصياغة أو الموضوع المصرفي بين بعض النوايا في هذه الأخطاء."
        ),

        "sec_text_length": "خصائص طول النصوص",
        "text_length_finding": (
            "ما طول النصوص في كل تقسيم؟ النسبة المئوية 95 توضح الطول الذي "
            "لا يتجاوزه 95% من النصوص: 5% فقط أطول."
        ),

        "methodology_title": "المنهجية",
        "meth_dataset": "**البيانات** : 10,732 تدريب / 1,229 تحقق / 3,580 اختبار سعودي",
        "meth_model": "**الموديل** : AraBERTv2 (aubmindlab/bert-base-arabertv2)",
        "meth_checkpoint": (
            "**نموذج العرض** : E1 / seed 2026، اختير بأعلى MSA validation F1 "
            "قبل رؤية بيانات الاختبار السعودي"
        ),
        "meth_conditions": "**الإعدادات التجريبية** : 5 (E0-E3 + EB) x 3 بذور (42, 123, 2026) = 15 تجربة",
        "meth_transfer": "**تقييم انتقال عبر اللهجات** : الفصحى ← السعودية",
        "meth_retrieval": (
            "**استرجاع تشخيصي** : يقارن مشفّر جمل متعدد اللغات كل مدخل "
            "بأمثلة التدريب المعتمدة بالفصحى. لا يؤثر ذلك في قرار AraBERT."
        ),

        "limitations_title": "القيود",
        "lim_calibration": "الدرجات مخرجات softmax، ليست احتمالات معايرة",
        "lim_oos": "لا كشف للنوايا خارج النطاق: النوايا غير المعروفة تُصنَّف رغم ذلك",
        "lim_routing": "لا قاعدة تلقائية للتصعيد عند عدم وضوح النتيجة",
        "lim_ensemble": "نقطة واحدة، بدون ensemble",
        "lim_privacy": (
            "تُعالج مدخلات المستخدم داخل الذاكرة، ولا يتضمن هذا النموذج الأولي "
            "قاعدة بيانات أو تخزينًا في ملفات لاستفسارات المستخدم."
        ),

        "future_work_title": "العمل المستقبلي",
        "future_work_1": (
            "معايرة درجات الثقة ووضع سياسة امتناع أو تحويل للمراجعة البشرية عند "
            "انخفاض الثقة."
        ),
        "future_work_2": (
            "إضافة كشف للطلبات خارج نطاق النوايا أو النوايا غير المعروفة، حتى لا "
            "تُفرض الطلبات غير المدعومة على واحدة من النوايا الـ77."
        ),
        "future_work_3": (
            "توسيع وتدقيق أمثلة مصرفية سعودية، خصوصًا أزواج النوايا كثيرة الالتباس، "
            "وتقييم أي دمج للنوايا عبر بروتوكول إعادة تدريب موثق."
        ),
        "future_work_4": (
            "مقارنة أساليب تكييف موفرة للمعلمات مثل LoRA، وCPT أكبر بضوابط واضحة، "
            "وEnsembles باستخدام بروتوكول الاختبار المحجوب نفسه."
        ),

        "footer": "صَرْف v1.0 · سبتمبر 2026 · نموذج أولي للاستدلال فقط",

        "no_data": "لا توجد بيانات.",
        "f1_label": "F1",
        "precision_label": "الدقة",
        "recall_label": "الاسترجاع",
        "support_label": "عدد العينات",

        "tip_understood": (
            "النموذج يرتب 77 نية مصرفية ويختار الأعلى درجة. "
            "النسبة المئوية هي درجة softmax: تُظهر تفضيل النموذج النسبي بين النوايا المتاحة، "
            "ولا تمثل احتمالًا مضمونًا للصحة."
        ),
        "tip_decision": (
            "يعرض الفارق بين درجتي softmax الأولى والثانية. "
            "فارق كبير يعني أن النموذج رتّب النية الأولى أعلى بكثير من الثانية لهذا المدخل؛ "
            "ذلك لا يضمن أن التوقع صحيح."
        ),
        "tip_alternatives": (
            "الترتيب الكامل لجميع النوايا الـ77 حسب درجة softmax. "
            "تُعرض أعلى 5 فقط. مجموع الدرجات لجميع النوايا يساوي 100%."
        ),
        "tip_confusion": (
            "يعرض هذا القسم أزواج الالتباس عالية التكرار في الاختبار السعودي "
            "المتعلقة بالنية المتوقعة. إذا لم يظهر أي زوج، فذلك يعني عدم وروده "
            "في قائمة حالات الالتباس الأعلى المحفوظة؛ ولا يثبت أن النية لم تلتبس أبدًا."
        ),
        "tip_retrieval": (
            "تشابه جيب التمام بين جملة المدخل وأمثلة من بيانات التدريب المعتمدة بالفصحى، "
            "محسوبة بمشفّر جمل متعدد اللغات. "
            "درجة أعلى تعني صياغة دلالية أقرب. "
            "هذا تشخيصي فقط ولا يحدد قرار AraBERT."
        ),
        "tip_model_compare": (
            "كل عمود يعرض متوسط Saudi Macro F1 عبر 3 بذور عشوائية (42, 123, 2026). "
            "الإعدادات التجريبية لـ AraBERT (E0-E3, EB) تختلف في كمية بيانات CPT "
            "السعودية الاصطناعية."
        ),
        "tip_f1_dist": (
            "درجات F1 لكل فئة لنموذج العرض (E1, seed 2026) على الاختبار السعودي. "
            "المدرج التكراري يوضح عدد النوايا في كل نطاق F1. "
            "الخط المتقطع يمثل المتوسط."
        ),
        "tip_confusions": (
            "أكثر أزواج التصنيف الخاطئ (الحقيقي → المتوقَّع) تكرارًا في الاختبار السعودي. "
            "الأعداد العالية تشير إلى نوايا متقاربة دلاليًا."
        ),
        "tip_intent_dist": (
            "عدد أمثلة التدريب بالفصحى لكل نية. عدم التوازن قد يؤدي إلى "
            "انخفاض F1 للنوايا قليلة التمثيل."
        ),
        "tip_text_length": (
            "إحصائيات عدد الكلمات لكل تقسيم بيانات: تدريب فصحى، تحقق فصحى، "
            "اختبار فصحى، واختبار سعودي. النسبة المئوية 95 توضح الطول الذي "
            "لا يتجاوزه 95% من النصوص."
        ),
    },
}
