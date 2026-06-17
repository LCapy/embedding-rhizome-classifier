from typing import List

TAXONOMY: List[dict] = [
    {
        "name": "Natural World",
        "level": 0,
        "parent": None,
        "en_category": "Natural sciences",
        "description": "Physical nature, life, earth systems, and the cosmos",
        "lang_categories": {
            "ru": "Естественные науки",
            "es": "Ciencias naturales",
            "fr": "Sciences naturelles",
            "pt": "Ciências naturais",
            "zh": "自然科学",
            "ar": "علوم طبيعية",
            "de": "Naturwissenschaft",
            "ja": "自然科学",
            "uk": "Природничі науки"
        },
        "related": [
            "Human Mind & Knowledge",
            "Human Activity & Society"
        ]
    },
    {
        "name": "Human Mind & Knowledge",
        "level": 0,
        "parent": None,
        "en_category": "Humanities",
        "description": "Thought, theory, abstract systems, and reflective inquiry",
        "lang_categories": {
            "ru": "Гуманитарные науки",
            "es": "Humanidades",
            "fr": "Sciences humaines",
            "pt": "Humanidades",
            "zh": "人文学科",
            "ar": "العلوم الإنسانية",
            "de": "Geisteswissenschaften",
            "ja": "人文科学",
            "uk": "Гуманітарні науки"
        },
        "related": [
            "Natural World",
            "Communication & Expression"
        ]
    },
    {
        "name": "Human Activity & Society",
        "level": 0,
        "parent": None,
        "en_category": "Society",
        "description": "Everyday actions, institutions, work, family, politics, and collective life",
        "lang_categories": {
            "ru": "Общество",
            "es": "Sociedad",
            "fr": "Société",
            "pt": "Sociedade",
            "zh": "社会",
            "ar": "مجتمع",
            "de": "Gesellschaft",
            "ja": "社会",
            "uk": "Суспільство"
        },
        "related": [
            "Human Mind & Knowledge",
            "Communication & Expression"
        ]
    },
    {
        "name": "Communication & Expression",
        "level": 0,
        "parent": None,
        "en_category": "Culture",
        "description": "Language, discourse, interaction, literature, and artistic expression",
        "lang_categories": {
            "ru": "Культура",
            "es": "Cultura",
            "fr": "Culture",
            "pt": "Cultura",
            "zh": "文化",
            "ar": "ثقافة",
            "de": "Kultur",
            "ja": "文化",
            "uk": "Культура"
        },
        "related": [
            "Human Mind & Knowledge",
            "Human Activity & Society"
        ]
    },
    {
        "name": "Physical Reality",
        "level": 1,
        "parent": "Natural World",
        "en_category": "Physical sciences",
        "description": "Matter, energy, forces, substances, and physical law",
        "lang_categories": {
            "ru": "Физические науки",
            "es": "Ciencias físicas",
            "fr": "Sciences physiques",
            "pt": "Ciências físicas",
            "zh": "物理科学",
            "de": "Physikalische Wissenschaften",
            "ja": "物理科学"
        },
        "related": [
            "Life & Biology",
            "Formal Systems"
        ]
    },
    {
        "name": "Life & Biology",
        "level": 1,
        "parent": "Natural World",
        "en_category": "Life sciences",
        "description": "Living systems, organisms, medicine, and ecology",
        "lang_categories": {
            "ru": "Науки о жизни",
            "es": "Ciencias de la vida",
            "fr": "Sciences de la vie",
            "pt": "Ciências da vida",
            "zh": "生命科学",
            "de": "Lebenswissenschaften",
            "ja": "生命科学"
        },
        "related": [
            "Physical Reality",
            "Daily Life"
        ]
    },
    {
        "name": "Earth & Cosmos",
        "level": 1,
        "parent": "Natural World",
        "en_category": "Earth sciences",
        "description": "Planetary systems, climate, geology, astronomy, and the universe",
        "lang_categories": {
            "ru": "Науки о Земле",
            "es": "Ciencias de la Tierra",
            "fr": "Sciences de la Terre",
            "pt": "Ciências da Terra",
            "zh": "地球科学",
            "de": "Geowissenschaften",
            "ja": "地球科学"
        },
        "related": [
            "Physical Reality",
            "Narrative & Reportage"
        ]
    },
    {
        "name": "Philosophy",
        "level": 1,
        "parent": "Human Mind & Knowledge",
        "en_category": "Philosophy",
        "description": "Reflection on reality, knowledge, values, and reason",
        "lang_categories": {
            "ru": "Философия",
            "es": "Filosofía",
            "fr": "Philosophie",
            "pt": "Filosofia",
            "zh": "哲学",
            "ar": "فلسفة",
            "de": "Philosophie",
            "ja": "哲学",
            "uk": "Філософія"
        },
        "related": [
            "Formal Systems",
            "Social Theory"
        ]
    },
    {
        "name": "Formal Systems",
        "level": 1,
        "parent": "Human Mind & Knowledge",
        "en_category": "Mathematics",
        "description": "Logic, mathematics, proof, abstraction, and symbolic structure",
        "lang_categories": {
            "ru": "Математика",
            "es": "Matemáticas",
            "fr": "Mathématiques",
            "pt": "Matemática",
            "zh": "数学",
            "ar": "رياضيات",
            "de": "Mathematik",
            "ja": "数学",
            "uk": "Математика"
        },
        "related": [
            "Physical Reality",
            "Philosophy"
        ]
    },
    {
        "name": "Social Theory",
        "level": 1,
        "parent": "Human Mind & Knowledge",
        "en_category": "Social sciences",
        "description": "Abstract models of society, economy, politics, and institutions",
        "lang_categories": {
            "ru": "Общественные науки",
            "es": "Ciencias sociales",
            "fr": "Sciences sociales",
            "pt": "Ciências sociais",
            "zh": "社会科学",
            "ar": "العلوم الاجتماعية",
            "de": "Sozialwissenschaften",
            "ja": "社会科学",
            "uk": "Соціальні науки"
        },
        "related": [
            "Institutions & Governance",
            "Work & Economy"
        ]
    },
    {
        "name": "Daily Life",
        "level": 1,
        "parent": "Human Activity & Society",
        "en_category": "Everyday life",
        "description": "Ordinary routines, home life, food, movement, and personal tasks",
        "lang_categories": {
            "ru": "Повседневная жизнь",
            "es": "Vida cotidiana",
            "fr": "Vie quotidienne",
            "pt": "Vida cotidiana",
            "zh": "日常生活",
            "de": "Alltagsleben",
            "ja": "日常生活"
        },
        "related": [
            "Social Relations",
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Work & Economy",
        "level": 1,
        "parent": "Human Activity & Society",
        "en_category": "Economy",
        "description": "Labor, professions, transactions, business, and services",
        "lang_categories": {
            "ru": "Экономика",
            "es": "Economía",
            "fr": "Économie",
            "pt": "Economia",
            "zh": "经济",
            "de": "Wirtschaft",
            "ja": "経済"
        },
        "related": [
            "Daily Life",
            "Institutions & Governance"
        ]
    },
    {
        "name": "Institutions & Governance",
        "level": 1,
        "parent": "Human Activity & Society",
        "en_category": "Government",
        "description": "States, laws, elections, public policy, and administration",
        "lang_categories": {
            "ru": "Государственное управление",
            "es": "Gobierno",
            "fr": "Gouvernement",
            "pt": "Governo",
            "zh": "政府",
            "de": "Regierung",
            "ja": "政府"
        },
        "related": [
            "Politics",
            "Law & Regulation",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Social Relations",
        "level": 1,
        "parent": "Human Activity & Society",
        "en_category": "Interpersonal relationships",
        "description": "Family, friendship, romance, community, and social bonds",
        "lang_categories": {
            "ru": "Межличностные отношения",
            "es": "Relaciones interpersonales",
            "fr": "Relations interpersonnelles",
            "pt": "Relações interpessoais",
            "zh": "人际关系",
            "de": "Zwischenmenschliche Beziehungen",
            "ja": "人間関係"
        },
        "related": [
            "Daily Life",
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Conflict & Cooperation",
        "level": 1,
        "parent": "Human Activity & Society",
        "en_category": "Conflict",
        "description": "Disagreement, negotiation, collaboration, and collective coordination",
        "lang_categories": {
            "ru": "Конфликт",
            "es": "Conflicto",
            "fr": "Conflit",
            "pt": "Conflito",
            "zh": "冲突",
            "de": "Konflikt",
            "ja": "対立"
        },
        "related": [
            "Social Relations",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Language Structure",
        "level": 1,
        "parent": "Communication & Expression",
        "en_category": "Linguistics",
        "description": "Grammar, meaning, use, and linguistic form",
        "lang_categories": {
            "ru": "Языкознание",
            "es": "Lingüística",
            "fr": "Linguistique",
            "pt": "Linguística",
            "zh": "语言学",
            "ar": "لغويات",
            "de": "Linguistik",
            "ja": "言語学",
            "uk": "Лінгвістика"
        },
        "related": [
            "Conversation & Dialogue",
            "Philosophy"
        ]
    },
    {
        "name": "Discourse Forms",
        "level": 1,
        "parent": "Communication & Expression",
        "en_category": "Discourse",
        "description": "Conversation, explanation, narrative, instruction, and persuasion",
        "lang_categories": {
            "ru": "Дискурс",
            "es": "Discurso",
            "fr": "Discours",
            "pt": "Discurso",
            "zh": "话语",
            "de": "Diskurs",
            "ja": "談話"
        },
        "related": [
            "Conversation & Dialogue",
            "Narrative & Reportage",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Creative Expression",
        "level": 1,
        "parent": "Communication & Expression",
        "en_category": "The arts",
        "description": "Literature, music, visual art, and expressive culture",
        "lang_categories": {
            "ru": "Искусство",
            "es": "Arte",
            "fr": "Art",
            "pt": "Arte",
            "zh": "艺术",
            "ar": "فن",
            "de": "Kunst",
            "ja": "芸術",
            "uk": "Мистецтво"
        },
        "related": [
            "Narrative & Reportage",
            "Language Structure"
        ]
    },
    {
        "name": "Physics",
        "level": 2,
        "parent": "Physical Reality",
        "en_category": "Physics",
        "description": "Matter, force, motion, energy, and physical law",
        "lang_categories": {
            "ru": "Физика",
            "es": "Física",
            "fr": "Physique",
            "pt": "Física",
            "zh": "物理学",
            "de": "Physik",
            "ja": "物理学"
        },
        "related": [
            "Chemistry",
            "Mathematics"
        ]
    },
    {
        "name": "Chemistry",
        "level": 2,
        "parent": "Physical Reality",
        "en_category": "Chemistry",
        "description": "Substances, reactions, molecules, and material composition",
        "lang_categories": {
            "ru": "Химия",
            "es": "Química",
            "fr": "Chimie",
            "pt": "Química",
            "zh": "化学",
            "de": "Chemie",
            "ja": "化学"
        },
        "related": [
            "Physics",
            "Biology"
        ]
    },
    {
        "name": "Biology",
        "level": 2,
        "parent": "Life & Biology",
        "en_category": "Biology",
        "description": "Living organisms, heredity, evolution, and biological systems",
        "lang_categories": {
            "ru": "Биология",
            "es": "Biología",
            "fr": "Biologie",
            "pt": "Biologia",
            "zh": "生物学",
            "de": "Biologie",
            "ja": "生物学"
        },
        "related": [
            "Medicine",
            "Ecology"
        ]
    },
    {
        "name": "Medicine",
        "level": 2,
        "parent": "Life & Biology",
        "en_category": "Medicine",
        "description": "Health, disease, diagnosis, treatment, and care",
        "lang_categories": {
            "ru": "Медицина",
            "es": "Medicina",
            "fr": "Médecine",
            "pt": "Medicina",
            "zh": "医学",
            "de": "Medizin",
            "ja": "医学"
        },
        "related": [
            "Biology",
            "Health Routine"
        ]
    },
    {
        "name": "Ecology",
        "level": 2,
        "parent": "Life & Biology",
        "en_category": "Ecology",
        "description": "Organisms in relation to environment and ecosystems",
        "lang_categories": {
            "ru": "Экология",
            "es": "Ecología",
            "fr": "Écologie",
            "pt": "Ecologia",
            "zh": "生态学",
            "de": "Ökologie",
            "ja": "生態学"
        },
        "related": [
            "Climate & Weather",
            "Biology"
        ]
    },
    {
        "name": "Geology",
        "level": 2,
        "parent": "Earth & Cosmos",
        "en_category": "Geology",
        "description": "Earth materials, rocks, tectonics, and planetary structure",
        "lang_categories": {
            "ru": "Геология",
            "es": "Geología",
            "fr": "Géologie",
            "pt": "Geologia",
            "zh": "地质学",
            "de": "Geologie",
            "ja": "地質学"
        },
        "related": [
            "Climate & Weather",
            "Astronomy"
        ]
    },
    {
        "name": "Climate & Weather",
        "level": 2,
        "parent": "Earth & Cosmos",
        "en_category": "Climatology",
        "description": "Atmospheric systems, weather processes, and climate patterns",
        "lang_categories": {
            "ru": "Климатология",
            "es": "Climatología",
            "fr": "Climatologie",
            "pt": "Climatologia",
            "zh": "气候学",
            "de": "Klimatologie",
            "ja": "気候学"
        },
        "related": [
            "Ecology",
            "Geology"
        ]
    },
    {
        "name": "Astronomy",
        "level": 2,
        "parent": "Earth & Cosmos",
        "en_category": "Astronomy",
        "description": "Celestial bodies, space, and the large-scale universe",
        "lang_categories": {
            "ru": "Астрономия",
            "es": "Astronomía",
            "fr": "Astronomie",
            "pt": "Astronomia",
            "zh": "天文学",
            "de": "Astronomie",
            "ja": "天文学"
        },
        "related": [
            "Cosmology",
            "Astrophysics"
        ]
    },
    {
        "name": "Ethics",
        "level": 2,
        "parent": "Philosophy",
        "en_category": "Ethics",
        "description": "Moral thought, values, and right action",
        "lang_categories": {
            "ru": "Этика",
            "es": "Ética",
            "fr": "Éthique",
            "pt": "Ética",
            "zh": "伦理学",
            "de": "Ethik",
            "ja": "倫理学"
        },
        "related": [
            "Law & Regulation",
            "Social Norms"
        ]
    },
    {
        "name": "Epistemology",
        "level": 2,
        "parent": "Philosophy",
        "en_category": "Epistemology",
        "description": "Knowledge, evidence, truth, and belief",
        "lang_categories": {
            "ru": "Эпистемология",
            "es": "Epistemología",
            "fr": "Épistémologie",
            "pt": "Epistemologia",
            "zh": "认识论",
            "de": "Erkenntnistheorie",
            "ja": "認識論"
        },
        "related": [
            "Logic",
            "Semantics"
        ]
    },
    {
        "name": "Logic",
        "level": 2,
        "parent": "Philosophy",
        "en_category": "Logic",
        "description": "Inference, validity, propositions, and reasoning patterns",
        "lang_categories": {
            "ru": "Логика",
            "es": "Lógica",
            "fr": "Logique",
            "pt": "Lógica",
            "zh": "逻辑学",
            "de": "Logik",
            "ja": "論理学"
        },
        "related": [
            "Mathematics",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Mathematics",
        "level": 2,
        "parent": "Formal Systems",
        "en_category": "Mathematics",
        "description": "Structure, quantity, pattern, and formal abstraction",
        "lang_categories": {
            "ru": "Математика",
            "es": "Matemáticas",
            "fr": "Mathématiques",
            "pt": "Matemática",
            "zh": "数学",
            "de": "Mathematik",
            "ja": "数学"
        },
        "related": [
            "Physics",
            "Logic"
        ]
    },
    {
        "name": "Statistics",
        "level": 2,
        "parent": "Formal Systems",
        "en_category": "Statistics",
        "description": "Data, uncertainty, variation, and inference",
        "lang_categories": {
            "ru": "Статистика",
            "es": "Estadística",
            "fr": "Statistique",
            "pt": "Estatística",
            "zh": "统计学",
            "de": "Statistik",
            "ja": "統計学"
        },
        "related": [
            "Mathematics",
            "Social Theory"
        ]
    },
    {
        "name": "Economics",
        "level": 2,
        "parent": "Social Theory",
        "en_category": "Economics",
        "description": "Production, exchange, incentives, and allocation",
        "lang_categories": {
            "ru": "Экономика",
            "es": "Economía",
            "fr": "Économie",
            "pt": "Economia",
            "zh": "经济学",
            "de": "Wirtschaftswissenschaft",
            "ja": "経済学"
        },
        "related": [
            "Work & Economy",
            "Politics"
        ]
    },
    {
        "name": "Politics",
        "level": 2,
        "parent": "Social Theory",
        "en_category": "Political science",
        "description": "Power, government, elections, and political systems",
        "lang_categories": {
            "ru": "Политология",
            "es": "Ciencia política",
            "fr": "Science politique",
            "pt": "Ciência política",
            "zh": "政治学",
            "de": "Politikwissenschaft",
            "ja": "政治学"
        },
        "related": [
            "Institutions & Governance",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Sociology",
        "level": 2,
        "parent": "Social Theory",
        "en_category": "Sociology",
        "description": "Groups, norms, institutions, and social structure",
        "lang_categories": {
            "ru": "Социология",
            "es": "Sociología",
            "fr": "Sociologie",
            "pt": "Sociologia",
            "zh": "社会学",
            "de": "Soziologie",
            "ja": "社会学"
        },
        "related": [
            "Social Relations",
            "Conflict & Cooperation"
        ]
    },
    {
        "name": "Food & Cooking",
        "level": 2,
        "parent": "Daily Life",
        "en_category": "Cooking",
        "description": "Preparing, consuming, and discussing food and meals",
        "lang_categories": {
            "ru": "Кулинария",
            "es": "Cocina",
            "fr": "Cuisine",
            "pt": "Culinária",
            "zh": "烹饪",
            "de": "Kochen",
            "ja": "料理"
        },
        "related": [
            "Home Life",
            "Shopping & Consumption"
        ]
    },
    {
        "name": "Shopping & Consumption",
        "level": 2,
        "parent": "Daily Life",
        "en_category": "Shopping",
        "description": "Buying goods, comparing products, and consumer decisions",
        "lang_categories": {
            "ru": "Покупки",
            "es": "Compras",
            "fr": "Achats",
            "pt": "Compras",
            "zh": "购物",
            "de": "Einkaufen",
            "ja": "買い物"
        },
        "related": [
            "Transactions",
            "Customer Service"
        ]
    },
    {
        "name": "Transportation & Travel",
        "level": 2,
        "parent": "Daily Life",
        "en_category": "Transport",
        "description": "Movement, commuting, trips, routes, and travel plans",
        "lang_categories": {
            "ru": "Транспорт",
            "es": "Transporte",
            "fr": "Transport",
            "pt": "Transporte",
            "zh": "交通",
            "de": "Verkehr",
            "ja": "交通"
        },
        "related": [
            "Planning & Scheduling",
            "Public Services"
        ]
    },
    {
        "name": "Health Routine",
        "level": 2,
        "parent": "Daily Life",
        "en_category": "Health",
        "description": "Self-care, symptoms, appointments, medication, and wellbeing",
        "lang_categories": {
            "ru": "Здоровье",
            "es": "Salud",
            "fr": "Santé",
            "pt": "Saúde",
            "zh": "健康",
            "de": "Gesundheit",
            "ja": "健康"
        },
        "related": [
            "Medicine",
            "Home Life"
        ]
    },
    {
        "name": "Home Life",
        "level": 2,
        "parent": "Daily Life",
        "en_category": "Home",
        "description": "Household tasks, home management, and domestic life",
        "lang_categories": {
            "ru": "Домашняя жизнь",
            "es": "Vida doméstica",
            "fr": "Vie domestique",
            "pt": "Vida doméstica",
            "zh": "家庭生活",
            "de": "Haushalt",
            "ja": "家庭生活"
        },
        "related": [
            "Family Life",
            "Food & Cooking"
        ]
    },
    {
        "name": "Jobs & Professions",
        "level": 2,
        "parent": "Work & Economy",
        "en_category": "Occupations",
        "description": "Roles, careers, workplace identity, and professional tasks",
        "lang_categories": {
            "ru": "Профессии",
            "es": "Profesiones",
            "fr": "Professions",
            "pt": "Profissões",
            "zh": "职业",
            "de": "Berufe",
            "ja": "職業"
        },
        "related": [
            "Workplace Coordination",
            "Education & Training"
        ]
    },
    {
        "name": "Business Operations",
        "level": 2,
        "parent": "Work & Economy",
        "en_category": "Business",
        "description": "Managing work, processes, clients, teams, and organizational tasks",
        "lang_categories": {
            "ru": "Бизнес",
            "es": "Negocios",
            "fr": "Affaires",
            "pt": "Negócios",
            "zh": "商业",
            "de": "Wirtschaft",
            "ja": "ビジネス"
        },
        "related": [
            "Customer Service",
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Transactions",
        "level": 2,
        "parent": "Work & Economy",
        "en_category": "Commerce",
        "description": "Payment, billing, selling, ordering, and exchange",
        "lang_categories": {
            "ru": "Торговля",
            "es": "Comercio",
            "fr": "Commerce",
            "pt": "Comércio",
            "zh": "商业交易",
            "de": "Handel",
            "ja": "取引"
        },
        "related": [
            "Shopping & Consumption",
            "Customer Service"
        ]
    },
    {
        "name": "Customer Service",
        "level": 2,
        "parent": "Work & Economy",
        "en_category": "Customer service",
        "description": "Support, complaints, assistance, troubleshooting, and service dialogue",
        "lang_categories": {
            "ru": "Обслуживание клиентов",
            "es": "Atención al cliente",
            "fr": "Service client",
            "pt": "Atendimento ao cliente",
            "zh": "客户服务",
            "de": "Kundendienst",
            "ja": "顧客対応"
        },
        "related": [
            "Conversation & Dialogue",
            "Transactions"
        ]
    },
    {
        "name": "Government",
        "level": 2,
        "parent": "Institutions & Governance",
        "en_category": "Government",
        "description": "Public authority, state administration, and official bodies",
        "lang_categories": {
            "ru": "Правительство",
            "es": "Gobierno",
            "fr": "Gouvernement",
            "pt": "Governo",
            "zh": "政府",
            "de": "Regierung",
            "ja": "政府"
        },
        "related": [
            "Politics",
            "Public Services"
        ]
    },
    {
        "name": "Law & Regulation",
        "level": 2,
        "parent": "Institutions & Governance",
        "en_category": "Law",
        "description": "Rules, legal systems, obligations, rights, and enforcement",
        "lang_categories": {
            "ru": "Право",
            "es": "Derecho",
            "fr": "Droit",
            "pt": "Direito",
            "zh": "法律",
            "de": "Recht",
            "ja": "法律"
        },
        "related": [
            "Government",
            "Ethics"
        ]
    },
    {
        "name": "Public Policy",
        "level": 2,
        "parent": "Institutions & Governance",
        "en_category": "Public policy",
        "description": "Programs, reforms, legislation, and policy proposals",
        "lang_categories": {
            "ru": "Государственная политика",
            "es": "Política pública",
            "fr": "Politique publique",
            "pt": "Política pública",
            "zh": "公共政策",
            "de": "Öffentliche Politik",
            "ja": "公共政策"
        },
        "related": [
            "Politics",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Elections & Campaigns",
        "level": 2,
        "parent": "Institutions & Governance",
        "en_category": "Elections",
        "description": "Voting, candidates, campaigns, and electoral processes",
        "lang_categories": {
            "ru": "Выборы",
            "es": "Elecciones",
            "fr": "Élections",
            "pt": "Eleições",
            "zh": "选举",
            "de": "Wahlen",
            "ja": "選挙"
        },
        "related": [
            "Politics",
            "Public Debate"
        ]
    },
    {
        "name": "Family Life",
        "level": 2,
        "parent": "Social Relations",
        "en_category": "Family",
        "description": "Parents, children, relatives, and domestic relationships",
        "lang_categories": {
            "ru": "Семья",
            "es": "Familia",
            "fr": "Famille",
            "pt": "Família",
            "zh": "家庭",
            "de": "Familie",
            "ja": "家族"
        },
        "related": [
            "Home Life",
            "Emotion & Social Bonding"
        ]
    },
    {
        "name": "Friendship & Community",
        "level": 2,
        "parent": "Social Relations",
        "en_category": "Friendship",
        "description": "Peers, friends, community ties, and belonging",
        "lang_categories": {
            "ru": "Дружба",
            "es": "Amistad",
            "fr": "Amitié",
            "pt": "Amizade",
            "zh": "友谊",
            "de": "Freundschaft",
            "ja": "友情"
        },
        "related": [
            "Conversation & Dialogue",
            "Conflict & Cooperation"
        ]
    },
    {
        "name": "Romance & Intimacy",
        "level": 2,
        "parent": "Social Relations",
        "en_category": "Love",
        "description": "Affection, partnership, emotional closeness, and personal bonds",
        "lang_categories": {
            "ru": "Любовь",
            "es": "Amor",
            "fr": "Amour",
            "pt": "Amor",
            "zh": "爱情",
            "de": "Liebe",
            "ja": "愛"
        },
        "related": [
            "Emotion & Social Bonding",
            "Family Life"
        ]
    },
    {
        "name": "Emotion & Social Bonding",
        "level": 2,
        "parent": "Social Relations",
        "en_category": "Emotion",
        "description": "Feelings, empathy, reassurance, support, and attachment",
        "lang_categories": {
            "ru": "Эмоции",
            "es": "Emoción",
            "fr": "Émotion",
            "pt": "Emoção",
            "zh": "情感",
            "de": "Emotion",
            "ja": "感情"
        },
        "related": [
            "Romance & Intimacy",
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Negotiation",
        "level": 2,
        "parent": "Conflict & Cooperation",
        "en_category": "Negotiation",
        "description": "Bargaining, compromise, terms, and strategic agreement",
        "lang_categories": {
            "ru": "Переговоры",
            "es": "Negociación",
            "fr": "Négociation",
            "pt": "Negociação",
            "zh": "谈判",
            "de": "Verhandlung",
            "ja": "交渉"
        },
        "related": [
            "Transactions",
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Disagreement & Argument",
        "level": 2,
        "parent": "Conflict & Cooperation",
        "en_category": "Argument",
        "description": "Opposition, criticism, rebuttal, and verbal conflict",
        "lang_categories": {
            "ru": "Спор",
            "es": "Discusión",
            "fr": "Dispute",
            "pt": "Discussão",
            "zh": "争论",
            "de": "Streit",
            "ja": "議論"
        },
        "related": [
            "Public Debate",
            "Family Discussion"
        ]
    },
    {
        "name": "Collaboration",
        "level": 2,
        "parent": "Conflict & Cooperation",
        "en_category": "Collaboration",
        "description": "Working together, shared goals, and coordinated effort",
        "lang_categories": {
            "ru": "Сотрудничество",
            "es": "Colaboración",
            "fr": "Collaboration",
            "pt": "Colaboração",
            "zh": "合作",
            "de": "Zusammenarbeit",
            "ja": "協力"
        },
        "related": [
            "Workplace Coordination",
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Competition",
        "level": 2,
        "parent": "Conflict & Cooperation",
        "en_category": "Competition",
        "description": "Rivalry, contests, strategic advantage, and winning",
        "lang_categories": {
            "ru": "Конкуренция",
            "es": "Competencia",
            "fr": "Compétition",
            "pt": "Competição",
            "zh": "竞争",
            "de": "Wettbewerb",
            "ja": "競争"
        },
        "related": [
            "Politics"
        ]
    },
    {
        "name": "Syntax",
        "level": 2,
        "parent": "Language Structure",
        "en_category": "Syntax",
        "description": "Sentence structure, ordering, and grammatical relations",
        "lang_categories": {
            "ru": "Синтаксис",
            "es": "Sintaxis",
            "fr": "Syntaxe",
            "pt": "Sintaxe",
            "zh": "句法",
            "de": "Syntax",
            "ja": "統語論"
        },
        "related": [
            "Semantics",
            "Pragmatics"
        ]
    },
    {
        "name": "Semantics",
        "level": 2,
        "parent": "Language Structure",
        "en_category": "Semantics",
        "description": "Meaning, reference, concepts, and interpretation",
        "lang_categories": {
            "ru": "Семантика",
            "es": "Semántica",
            "fr": "Sémantique",
            "pt": "Semântica",
            "zh": "语义学",
            "de": "Semantik",
            "ja": "意味論"
        },
        "related": [
            "Epistemology",
            "Pragmatics"
        ]
    },
    {
        "name": "Pragmatics",
        "level": 2,
        "parent": "Language Structure",
        "en_category": "Pragmatics",
        "description": "Use, intention, context, and communicative action",
        "lang_categories": {
            "ru": "Прагматика",
            "es": "Pragmática",
            "fr": "Pragmatique",
            "pt": "Pragmática",
            "zh": "语用学",
            "de": "Pragmatik",
            "ja": "語用論"
        },
        "related": [
            "Speech Acts",
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Conversation & Dialogue",
        "level": 2,
        "parent": "Discourse Forms",
        "en_category": "Conversation",
        "description": "Two or more speakers exchanging utterances in interaction",
        "lang_categories": {
            "ru": "Разговор",
            "es": "Conversación",
            "fr": "Conversation",
            "pt": "Conversa",
            "zh": "对话",
            "de": "Gespräch",
            "ja": "会話"
        },
        "related": [
            "Question–Answer",
            "Speech Acts"
        ]
    },
    {
        "name": "Instruction & Procedure",
        "level": 2,
        "parent": "Discourse Forms",
        "en_category": "Instruction",
        "description": "Guidance, steps, explanations of how to do something",
        "lang_categories": {
            "ru": "Инструкция",
            "es": "Instrucción",
            "fr": "Instruction",
            "pt": "Instrução",
            "zh": "说明",
            "de": "Anleitung",
            "ja": "手順"
        },
        "related": [
            "Planning & Scheduling",
            "Customer Service"
        ]
    },
    {
        "name": "Narrative & Reportage",
        "level": 2,
        "parent": "Discourse Forms",
        "en_category": "Narrative",
        "description": "Events, stories, reports, and accounts of what happened",
        "lang_categories": {
            "ru": "Повествование",
            "es": "Narrativa",
            "fr": "Narration",
            "pt": "Narrativa",
            "zh": "叙事",
            "de": "Erzählung",
            "ja": "物語"
        },
        "related": [
            "Creative Expression"
        ]
    },
    {
        "name": "Argument & Persuasion",
        "level": 2,
        "parent": "Discourse Forms",
        "en_category": "Argument",
        "description": "Claims, reasons, persuasion, rebuttal, and justification",
        "lang_categories": {
            "ru": "Аргументация",
            "es": "Argumentación",
            "fr": "Argumentation",
            "pt": "Argumentação",
            "zh": "论证",
            "de": "Argumentation",
            "ja": "議論"
        },
        "related": [
            "Politics",
            "Disagreement & Argument"
        ]
    },
    {
        "name": "Question–Answer",
        "level": 2,
        "parent": "Discourse Forms",
        "en_category": "Question",
        "description": "Inquiry, response, clarification, and information exchange",
        "lang_categories": {
            "ru": "Вопрос-ответ",
            "es": "Pregunta y respuesta",
            "fr": "Question-réponse",
            "pt": "Pergunta e resposta",
            "zh": "问答",
            "de": "Frage und Antwort",
            "ja": "質疑応答"
        },
        "related": [
            "Conversation & Dialogue",
            "Customer Service"
        ]
    },
    {
        "name": "Literature",
        "level": 2,
        "parent": "Creative Expression",
        "en_category": "Literature",
        "description": "Written imaginative or expressive language",
        "lang_categories": {
            "ru": "Литература",
            "es": "Literatura",
            "fr": "Littérature",
            "pt": "Literatura",
            "zh": "文学",
            "de": "Literatur",
            "ja": "文学"
        },
        "related": [
            "Narrative & Reportage",
            "Poetry"
        ]
    },
    {
        "name": "Music",
        "level": 2,
        "parent": "Creative Expression",
        "en_category": "Music",
        "description": "Sound, rhythm, melody, song, and musical form",
        "lang_categories": {
            "ru": "Музыка",
            "es": "Música",
            "fr": "Musique",
            "pt": "Música",
            "zh": "音乐",
            "de": "Musik",
            "ja": "音楽"
        },
        "related": [
            "Performance",
            "Emotion & Social Bonding"
        ]
    },
    {
        "name": "Visual Arts",
        "level": 2,
        "parent": "Creative Expression",
        "en_category": "Visual arts",
        "description": "Painting, image, visual design, and artistic objects",
        "lang_categories": {
            "ru": "Изобразительное искусство",
            "es": "Artes visuales",
            "fr": "Arts visuels",
            "pt": "Artes visuais",
            "zh": "视觉艺术",
            "de": "Bildende Kunst",
            "ja": "視覚芸術"
        },
        "related": [
            "Literature",
            "Performance"
        ]
    },
    {
        "name": "Performance",
        "level": 2,
        "parent": "Creative Expression",
        "en_category": "Performing arts",
        "description": "Theater, enactment, staged action, and performance",
        "lang_categories": {
            "ru": "Исполнительское искусство",
            "es": "Artes escénicas",
            "fr": "Arts du spectacle",
            "pt": "Artes cênicas",
            "zh": "表演艺术",
            "de": "Darstellende Kunst",
            "ja": "舞台芸術"
        },
        "related": [
            "Music",
            "Dialogue Scene"
        ]
    },
    {
        "name": "Mechanics",
        "level": 3,
        "parent": "Physics",
        "en_category": "Mechanics",
        "description": "Motion, bodies, force, and mechanical systems",
        "lang_categories": {},
        "related": [
            "Thermodynamics"
        ]
    },
    {
        "name": "Thermodynamics",
        "level": 3,
        "parent": "Physics",
        "en_category": "Thermodynamics",
        "description": "Heat, energy transfer, work, and entropy",
        "lang_categories": {},
        "related": [
            "Mechanics"
        ]
    },
    {
        "name": "Quantum Mechanics",
        "level": 3,
        "parent": "Physics",
        "en_category": "Quantum mechanics",
        "description": "Microscopic systems, wave functions, and quantization",
        "lang_categories": {},
        "related": [
            "Mathematics"
        ]
    },
    {
        "name": "Organic Chemistry",
        "level": 3,
        "parent": "Chemistry",
        "en_category": "Organic chemistry",
        "description": "Carbon compounds and molecular transformations",
        "lang_categories": {},
        "related": [
            "Biology"
        ]
    },
    {
        "name": "Inorganic Chemistry",
        "level": 3,
        "parent": "Chemistry",
        "en_category": "Inorganic chemistry",
        "description": "Non-organic compounds, materials, and reactions",
        "lang_categories": {},
        "related": [
            "Physics"
        ]
    },
    {
        "name": "Genetics",
        "level": 3,
        "parent": "Biology",
        "en_category": "Genetics",
        "description": "Genes, heredity, and biological inheritance",
        "lang_categories": {},
        "related": [
            "Evolution"
        ]
    },
    {
        "name": "Evolution",
        "level": 3,
        "parent": "Biology",
        "en_category": "Evolution",
        "description": "Change in populations over time and natural selection",
        "lang_categories": {},
        "related": [
            "Genetics",
            "Ecology"
        ]
    },
    {
        "name": "Diagnosis",
        "level": 3,
        "parent": "Medicine",
        "en_category": "Medical diagnosis",
        "description": "Symptoms, tests, identification, and clinical judgment",
        "lang_categories": {},
        "related": [
            "Treatment"
        ]
    },
    {
        "name": "Treatment",
        "level": 3,
        "parent": "Medicine",
        "en_category": "Therapy",
        "description": "Care, intervention, medication, and recovery",
        "lang_categories": {},
        "related": [
            "Diagnosis"
        ]
    },
    {
        "name": "Ecosystems",
        "level": 3,
        "parent": "Ecology",
        "en_category": "Ecosystem",
        "description": "Interacting organisms and their environments",
        "lang_categories": {},
        "related": [
            "Climate & Weather"
        ]
    },
    {
        "name": "Tectonics",
        "level": 3,
        "parent": "Geology",
        "en_category": "Plate tectonics",
        "description": "Earth structure, plates, and geological movement",
        "lang_categories": {},
        "related": [
            "Mechanics"
        ]
    },
    {
        "name": "Meteorology",
        "level": 3,
        "parent": "Climate & Weather",
        "en_category": "Meteorology",
        "description": "Weather systems, forecasting, and atmospheric conditions",
        "lang_categories": {},
        "related": [
            "Climate Change"
        ]
    },
    {
        "name": "Cosmology",
        "level": 3,
        "parent": "Astronomy",
        "en_category": "Cosmology",
        "description": "Origin, structure, and evolution of the universe",
        "lang_categories": {},
        "related": [
            "Astrophysics"
        ]
    },
    {
        "name": "Astrophysics",
        "level": 3,
        "parent": "Astronomy",
        "en_category": "Astrophysics",
        "description": "Physical processes governing celestial objects",
        "lang_categories": {},
        "related": [
            "Cosmology"
        ]
    },
    {
        "name": "Moral Philosophy",
        "level": 3,
        "parent": "Ethics",
        "en_category": "Moral philosophy",
        "description": "Theories of right and wrong",
        "lang_categories": {},
        "related": [
            "Law & Regulation"
        ]
    },
    {
        "name": "Knowledge & Belief",
        "level": 3,
        "parent": "Epistemology",
        "en_category": "Belief",
        "description": "Truth, evidence, certainty, and justified belief",
        "lang_categories": {},
        "related": [
            "Semantics"
        ]
    },
    {
        "name": "Proof & Inference",
        "level": 3,
        "parent": "Logic",
        "en_category": "Inference",
        "description": "Deduction, validity, and consequence",
        "lang_categories": {},
        "related": [
            "Mathematics"
        ]
    },
    {
        "name": "Algebra",
        "level": 3,
        "parent": "Mathematics",
        "en_category": "Algebra",
        "description": "Symbols, equations, and abstract operations",
        "lang_categories": {},
        "related": [
            "Geometry"
        ]
    },
    {
        "name": "Geometry",
        "level": 3,
        "parent": "Mathematics",
        "en_category": "Geometry",
        "description": "Shape, space, and spatial relations",
        "lang_categories": {},
        "related": [
            "Algebra"
        ]
    },
    {
        "name": "Probability & Inference",
        "level": 3,
        "parent": "Statistics",
        "en_category": "Statistical inference",
        "description": "Estimating, testing, and reasoning under uncertainty",
        "lang_categories": {},
        "related": [
            "Knowledge & Belief"
        ]
    },
    {
        "name": "Macroeconomics",
        "level": 3,
        "parent": "Economics",
        "en_category": "Macroeconomics",
        "description": "Inflation, unemployment, growth, and monetary systems",
        "lang_categories": {},
        "related": [
            "Public Policy"
        ]
    },
    {
        "name": "Microeconomics",
        "level": 3,
        "parent": "Economics",
        "en_category": "Microeconomics",
        "description": "Choice, incentives, prices, and local market behavior",
        "lang_categories": {},
        "related": [
            "Transactions"
        ]
    },
    {
        "name": "Political Institutions",
        "level": 3,
        "parent": "Politics",
        "en_category": "Political institutions",
        "description": "Legislatures, executives, parties, and governing structures",
        "lang_categories": {},
        "related": [
            "Government"
        ]
    },
    {
        "name": "Public Ideology",
        "level": 3,
        "parent": "Politics",
        "en_category": "Political ideologies",
        "description": "Competing visions of state, rights, and social order",
        "lang_categories": {},
        "related": [
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Social Norms",
        "level": 3,
        "parent": "Sociology",
        "en_category": "Social norm",
        "description": "Expected behavior, conventions, and collective practice",
        "lang_categories": {},
        "related": [
            "Family Life"
        ]
    },
    {
        "name": "Institutions as Systems",
        "level": 3,
        "parent": "Sociology",
        "en_category": "Social institution",
        "description": "Organizations, rules, and patterned social order",
        "lang_categories": {},
        "related": [
            "Government"
        ]
    },
    {
        "name": "Meal Preparation",
        "level": 3,
        "parent": "Food & Cooking",
        "en_category": "Cooking",
        "description": "Preparing ingredients, cooking food, and serving meals",
        "lang_categories": {},
        "related": [
            "Instruction & Procedure"
        ]
    },
    {
        "name": "Restaurant & Ordering",
        "level": 3,
        "parent": "Food & Cooking",
        "en_category": "Restaurant",
        "description": "Menus, ordering food, payment, and dining interactions",
        "lang_categories": {},
        "related": [
            "Transactions"
        ]
    },
    {
        "name": "Product Choice",
        "level": 3,
        "parent": "Shopping & Consumption",
        "en_category": "Consumer behaviour",
        "description": "Comparing options, selecting goods, and evaluating purchases",
        "lang_categories": {},
        "related": [
            "Recommendation Dialogue"
        ]
    },
    {
        "name": "Purchase & Payment",
        "level": 3,
        "parent": "Shopping & Consumption",
        "en_category": "Sales",
        "description": "Buying, paying, checkout, and receipt-related actions",
        "lang_categories": {},
        "related": [
            "Transactions"
        ]
    },
    {
        "name": "Commuting",
        "level": 3,
        "parent": "Transportation & Travel",
        "en_category": "Commuting",
        "description": "Routine movement to work, school, or appointments",
        "lang_categories": {},
        "related": [
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Trip Planning",
        "level": 3,
        "parent": "Transportation & Travel",
        "en_category": "Travel",
        "description": "Routes, schedules, destinations, and arrangements",
        "lang_categories": {},
        "related": [
            "Instruction & Procedure"
        ]
    },
    {
        "name": "Symptoms & Self-Care",
        "level": 3,
        "parent": "Health Routine",
        "en_category": "Self-care",
        "description": "Daily health, discomfort, rest, and self-management",
        "lang_categories": {},
        "related": [
            "Diagnosis"
        ]
    },
    {
        "name": "Appointments & Medication",
        "level": 3,
        "parent": "Health Routine",
        "en_category": "Medication",
        "description": "Doctor visits, prescriptions, and treatment adherence",
        "lang_categories": {},
        "related": [
            "Treatment"
        ]
    },
    {
        "name": "Household Maintenance",
        "level": 3,
        "parent": "Home Life",
        "en_category": "Housekeeping",
        "description": "Cleaning, fixing, organizing, and maintaining the home",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Domestic Planning",
        "level": 3,
        "parent": "Home Life",
        "en_category": "Household management",
        "description": "Managing chores, bills, and home coordination",
        "lang_categories": {},
        "related": [
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Workplace Coordination",
        "level": 3,
        "parent": "Jobs & Professions",
        "en_category": "Workplace",
        "description": "Assigning tasks, status updates, and professional collaboration",
        "lang_categories": {},
        "related": [
            "Collaboration"
        ]
    },
    {
        "name": "Education & Training",
        "level": 3,
        "parent": "Jobs & Professions",
        "en_category": "Training",
        "description": "Learning, coaching, onboarding, and skill development",
        "lang_categories": {},
        "related": [
            "Instruction & Procedure"
        ]
    },
    {
        "name": "Operations Management",
        "level": 3,
        "parent": "Business Operations",
        "en_category": "Operations management",
        "description": "Processes, planning, logistics, and organizational execution",
        "lang_categories": {},
        "related": [
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Client Communication",
        "level": 3,
        "parent": "Business Operations",
        "en_category": "Business communication",
        "description": "Service messages, client updates, and transactional coordination",
        "lang_categories": {},
        "related": [
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Billing & Refunds",
        "level": 3,
        "parent": "Transactions",
        "en_category": "Billing",
        "description": "Charges, invoices, reimbursements, and payment correction",
        "lang_categories": {},
        "related": [
            "Complaint Handling"
        ]
    },
    {
        "name": "Ordering & Delivery",
        "level": 3,
        "parent": "Transactions",
        "en_category": "Order fulfillment",
        "description": "Orders, shipping, arrival, and delivery issues",
        "lang_categories": {},
        "related": [
            "Customer Service"
        ]
    },
    {
        "name": "Complaint Handling",
        "level": 3,
        "parent": "Customer Service",
        "en_category": "Complaints",
        "description": "Problems reported by customers and service recovery",
        "lang_categories": {},
        "related": [
            "Billing & Refunds"
        ]
    },
    {
        "name": "Troubleshooting Support",
        "level": 3,
        "parent": "Customer Service",
        "en_category": "Technical support",
        "description": "Diagnosing problems, guidance, and issue resolution",
        "lang_categories": {},
        "related": [
            "Instruction & Procedure"
        ]
    },
    {
        "name": "Public Administration",
        "level": 3,
        "parent": "Government",
        "en_category": "Public administration",
        "description": "Agencies, offices, permits, and state procedures",
        "lang_categories": {},
        "related": [
            "Public Services"
        ]
    },
    {
        "name": "Public Services",
        "level": 3,
        "parent": "Government",
        "en_category": "Public services",
        "description": "Utilities, transport, health, and citizen-facing state services",
        "lang_categories": {},
        "related": [
            "Public Administration"
        ]
    },
    {
        "name": "Rights & Obligations",
        "level": 3,
        "parent": "Law & Regulation",
        "en_category": "Rights",
        "description": "Permissions, duties, compliance, and entitlement",
        "lang_categories": {},
        "related": [
            "Moral Philosophy"
        ]
    },
    {
        "name": "Rules & Enforcement",
        "level": 3,
        "parent": "Law & Regulation",
        "en_category": "Law enforcement",
        "description": "Rules, sanctions, compliance, and enforcement actions",
        "lang_categories": {},
        "related": [
            "Public Administration"
        ]
    },
    {
        "name": "Policy Proposal",
        "level": 3,
        "parent": "Public Policy",
        "en_category": "Policy",
        "description": "Suggested reforms, legislation, and state intervention plans",
        "lang_categories": {},
        "related": [
            "Public Debate"
        ]
    },
    {
        "name": "Policy Impact",
        "level": 3,
        "parent": "Public Policy",
        "en_category": "Policy analysis",
        "description": "Consequences, outcomes, and evaluation of policy",
        "lang_categories": {},
        "related": [
            "Macroeconomics"
        ]
    },
    {
        "name": "Campaign Messaging",
        "level": 3,
        "parent": "Elections & Campaigns",
        "en_category": "Political campaign",
        "description": "Slogans, promises, positioning, and voter targeting",
        "lang_categories": {},
        "related": [
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Voting Process",
        "level": 3,
        "parent": "Elections & Campaigns",
        "en_category": "Voting",
        "description": "Casting ballots, counting votes, and electoral procedures",
        "lang_categories": {},
        "related": [
            "Government"
        ]
    },
    {
        "name": "Parent–Child Interaction",
        "level": 3,
        "parent": "Family Life",
        "en_category": "Parenting",
        "description": "Guidance, care, discipline, and family communication",
        "lang_categories": {},
        "related": [
            "Instruction & Procedure"
        ]
    },
    {
        "name": "Family Discussion",
        "level": 3,
        "parent": "Family Life",
        "en_category": "Family",
        "description": "Domestic conversation, planning, support, and disagreement",
        "lang_categories": {},
        "related": [
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Casual Socializing",
        "level": 3,
        "parent": "Friendship & Community",
        "en_category": "Socialization",
        "description": "Informal interaction, small talk, and social engagement",
        "lang_categories": {},
        "related": [
            "Greeting Exchange"
        ]
    },
    {
        "name": "Community Coordination",
        "level": 3,
        "parent": "Friendship & Community",
        "en_category": "Community",
        "description": "Group organization, announcements, and mutual support",
        "lang_categories": {},
        "related": [
            "Collaboration"
        ]
    },
    {
        "name": "Affection & Reassurance",
        "level": 3,
        "parent": "Romance & Intimacy",
        "en_category": "Affection",
        "description": "Warmth, reassurance, closeness, and loving exchange",
        "lang_categories": {},
        "related": [
            "Emotion & Social Bonding"
        ]
    },
    {
        "name": "Relationship Tension",
        "level": 3,
        "parent": "Romance & Intimacy",
        "en_category": "Relationship counseling",
        "description": "Conflict, apology, insecurity, and emotional repair",
        "lang_categories": {},
        "related": [
            "Disagreement & Argument"
        ]
    },
    {
        "name": "Empathy & Support",
        "level": 3,
        "parent": "Emotion & Social Bonding",
        "en_category": "Empathy",
        "description": "Comforting, understanding, checking in, and emotional care",
        "lang_categories": {},
        "related": [
            "Health Routine"
        ]
    },
    {
        "name": "Emotion Expression",
        "level": 3,
        "parent": "Emotion & Social Bonding",
        "en_category": "Emotional expression",
        "description": "Saying how one feels or reacting affectively",
        "lang_categories": {},
        "related": [
            "Speech Acts"
        ]
    },
    {
        "name": "Bargaining",
        "level": 3,
        "parent": "Negotiation",
        "en_category": "Bargaining",
        "description": "Offers, counteroffers, and concession-making",
        "lang_categories": {},
        "related": [
            "Transactions"
        ]
    },
    {
        "name": "Agreement Formation",
        "level": 3,
        "parent": "Negotiation",
        "en_category": "Agreement",
        "description": "Reaching terms, confirming commitments, and settling differences",
        "lang_categories": {},
        "related": [
            "Collaboration"
        ]
    },
    {
        "name": "Verbal Dispute",
        "level": 3,
        "parent": "Disagreement & Argument",
        "en_category": "Dispute",
        "description": "Open disagreement, accusation, and oppositional exchange",
        "lang_categories": {},
        "related": [
            "Debate"
        ]
    },
    {
        "name": "Critique & Rebuttal",
        "level": 3,
        "parent": "Disagreement & Argument",
        "en_category": "Rebuttal",
        "description": "Responding against a claim with reasons or objections",
        "lang_categories": {},
        "related": [
            "Argument & Persuasion"
        ]
    },
    {
        "name": "Joint Planning",
        "level": 3,
        "parent": "Collaboration",
        "en_category": "Planning",
        "description": "Coordinating next steps, roles, and timing",
        "lang_categories": {},
        "related": [
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Task Cooperation",
        "level": 3,
        "parent": "Collaboration",
        "en_category": "Teamwork",
        "description": "Carrying out shared work across people",
        "lang_categories": {},
        "related": [
            "Workplace Coordination"
        ]
    },
    {
        "name": "Sentence Structure",
        "level": 3,
        "parent": "Syntax",
        "en_category": "Sentence structure",
        "description": "Order, constituency, and grammatical linkage",
        "lang_categories": {},
        "related": [
            "Grammar Explanation"
        ]
    },
    {
        "name": "Meaning Relations",
        "level": 3,
        "parent": "Semantics",
        "en_category": "Meaning",
        "description": "Reference, sense, ambiguity, and conceptual meaning",
        "lang_categories": {},
        "related": [
            "Knowledge & Belief"
        ]
    },
    {
        "name": "Speech Acts",
        "level": 3,
        "parent": "Pragmatics",
        "en_category": "Speech act",
        "description": "Utterances used to request, promise, apologize, and more",
        "lang_categories": {},
        "related": [
            "Conversation & Dialogue"
        ]
    },
    {
        "name": "Dialogue Scene",
        "level": 3,
        "parent": "Conversation & Dialogue",
        "en_category": "Dialogue",
        "description": "Two or more people speaking turn by turn",
        "lang_categories": {},
        "related": [
            "Question–Answer"
        ]
    },
    {
        "name": "Small Talk",
        "level": 3,
        "parent": "Conversation & Dialogue",
        "en_category": "Small talk",
        "description": "Light social exchange and informal conversation",
        "lang_categories": {},
        "related": [
            "Casual Socializing"
        ]
    },
    {
        "name": "How-to Guidance",
        "level": 3,
        "parent": "Instruction & Procedure",
        "en_category": "How-to",
        "description": "Procedural explanation of steps to achieve an outcome",
        "lang_categories": {},
        "related": [
            "Troubleshooting Support"
        ]
    },
    {
        "name": "Planning & Scheduling",
        "level": 3,
        "parent": "Instruction & Procedure",
        "en_category": "Scheduling",
        "description": "Setting times, deadlines, and ordered actions",
        "lang_categories": {},
        "related": [
            "Joint Planning"
        ]
    },
    {
        "name": "Event Narrative",
        "level": 3,
        "parent": "Narrative & Reportage",
        "en_category": "Narrative",
        "description": "Describing what happened over time",
        "lang_categories": {},
        "related": [
            "Personal Story"
        ]
    },
    {
        "name": "Factual Report",
        "level": 3,
        "parent": "Narrative & Reportage",
        "en_category": "Report",
        "description": "Structured account of events or states of affairs",
        "lang_categories": {},
        "related": [
            "News Report"
        ]
    },
    {
        "name": "Debate",
        "level": 3,
        "parent": "Argument & Persuasion",
        "en_category": "Debate",
        "description": "Competing positions argued in public or interpersonal exchange",
        "lang_categories": {},
        "related": [
            "Politics"
        ]
    },
    {
        "name": "Recommendation Dialogue",
        "level": 3,
        "parent": "Argument & Persuasion",
        "en_category": "Recommendation",
        "description": "Advising, suggesting, or persuading toward a choice",
        "lang_categories": {},
        "related": [
            "Product Choice"
        ]
    },
    {
        "name": "Clarification Exchange",
        "level": 3,
        "parent": "Question–Answer",
        "en_category": "Clarification",
        "description": "Questions and answers used to resolve uncertainty",
        "lang_categories": {},
        "related": [
            "Customer Service"
        ]
    },
    {
        "name": "Information Seeking",
        "level": 3,
        "parent": "Question–Answer",
        "en_category": "Information",
        "description": "Looking for facts, directions, explanation, or confirmation",
        "lang_categories": {},
        "related": [
            "How-to Guidance"
        ]
    },
    {
        "name": "Poetry",
        "level": 3,
        "parent": "Literature",
        "en_category": "Poetry",
        "description": "Verse, condensed expression, rhythm, and figurative language",
        "lang_categories": {},
        "related": [
            "Music"
        ]
    },
    {
        "name": "Fiction",
        "level": 3,
        "parent": "Literature",
        "en_category": "Fiction",
        "description": "Imagined stories, characters, and invented worlds",
        "lang_categories": {},
        "related": [
            "Event Narrative"
        ]
    },
    {
        "name": "Song",
        "level": 3,
        "parent": "Music",
        "en_category": "Song",
        "description": "Structured vocal musical expression",
        "lang_categories": {},
        "related": [
            "Poetry"
        ]
    },
    {
        "name": "Image-making",
        "level": 3,
        "parent": "Visual Arts",
        "en_category": "Painting",
        "description": "Visual representation and artistic depiction",
        "lang_categories": {},
        "related": [
            "Performance"
        ]
    },
    {
        "name": "Stage Interaction",
        "level": 3,
        "parent": "Performance",
        "en_category": "Theatre",
        "description": "Performative dialogue, action, and staged scenes",
        "lang_categories": {},
        "related": [
            "Dialogue Scene"
        ]
    },
    {
        "name": "Wave Function",
        "level": 4,
        "parent": "Quantum Mechanics",
        "en_category": "Wave function",
        "description": "Mathematical representation of quantum state",
        "lang_categories": {},
        "related": [
            "Probability & Inference"
        ]
    },
    {
        "name": "Entropy",
        "level": 4,
        "parent": "Thermodynamics",
        "en_category": "Entropy",
        "description": "Disorder, information, and thermodynamic state",
        "lang_categories": {},
        "related": [
            "Heat Transfer"
        ]
    },
    {
        "name": "DNA Inheritance",
        "level": 4,
        "parent": "Genetics",
        "en_category": "DNA",
        "description": "Transmission of hereditary information",
        "lang_categories": {},
        "related": [
            "Genes"
        ]
    },
    {
        "name": "Natural Selection",
        "level": 4,
        "parent": "Evolution",
        "en_category": "Natural selection",
        "description": "Differential survival and reproduction",
        "lang_categories": {},
        "related": [
            "Adaptation"
        ]
    },
    {
        "name": "Climate Change",
        "level": 3,
        "parent": "Climate & Weather",
        "en_category": "Climate change",
        "description": "Long-term shifts in temperature and weather systems",
        "lang_categories": {},
        "related": [
            "Policy Impact"
        ]
    },
    {
        "name": "Big Bang Model",
        "level": 4,
        "parent": "Cosmology",
        "en_category": "Big Bang",
        "description": "Model of the early universe and expansion",
        "lang_categories": {},
        "related": [
            "Universe Expansion"
        ]
    },
    {
        "name": "Ordering at a Restaurant",
        "level": 4,
        "parent": "Restaurant & Ordering",
        "en_category": "Restaurant",
        "description": "Choosing dishes, placing an order, and interacting with staff",
        "lang_categories": {},
        "related": [
            "Purchase & Payment"
        ]
    },
    {
        "name": "Grocery Purchase",
        "level": 4,
        "parent": "Purchase & Payment",
        "en_category": "Grocery store",
        "description": "Selecting everyday goods and paying for them",
        "lang_categories": {},
        "related": [
            "Product Choice"
        ]
    },
    {
        "name": "Commute Delay",
        "level": 4,
        "parent": "Commuting",
        "en_category": "Delay",
        "description": "Late arrival, transit disruption, and schedule impact",
        "lang_categories": {},
        "related": [
            "Planning & Scheduling"
        ]
    },
    {
        "name": "Doctor Appointment",
        "level": 4,
        "parent": "Appointments & Medication",
        "en_category": "Medical appointment",
        "description": "Seeing a doctor, discussing symptoms, and follow-up",
        "lang_categories": {},
        "related": [
            "Diagnosis"
        ]
    },
    {
        "name": "Home Repair Request",
        "level": 4,
        "parent": "Household Maintenance",
        "en_category": "Home repair",
        "description": "Reporting or arranging a household fix",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Task Assignment",
        "level": 4,
        "parent": "Workplace Coordination",
        "en_category": "Task",
        "description": "Giving work responsibilities and expected outcomes",
        "lang_categories": {},
        "related": [
            "Joint Planning"
        ]
    },
    {
        "name": "Status Update",
        "level": 4,
        "parent": "Workplace Coordination",
        "en_category": "Progress report",
        "description": "Reporting progress, blockers, and next steps",
        "lang_categories": {},
        "related": [
            "Factual Report"
        ]
    },
    {
        "name": "Double Charge Complaint",
        "level": 4,
        "parent": "Billing & Refunds",
        "en_category": "Billing",
        "description": "A customer reports being charged twice",
        "lang_categories": {},
        "related": [
            "Complaint Handling"
        ]
    },
    {
        "name": "Refund Processing",
        "level": 4,
        "parent": "Billing & Refunds",
        "en_category": "Refund",
        "description": "Correcting a billing error and returning payment",
        "lang_categories": {},
        "related": [
            "Agreement Formation"
        ]
    },
    {
        "name": "Late Delivery Problem",
        "level": 4,
        "parent": "Ordering & Delivery",
        "en_category": "Delivery",
        "description": "A missing or delayed order and the resulting support case",
        "lang_categories": {},
        "related": [
            "Complaint Handling"
        ]
    },
    {
        "name": "Technical Setup Help",
        "level": 4,
        "parent": "Troubleshooting Support",
        "en_category": "Technical support",
        "description": "Helping a user configure or fix a system",
        "lang_categories": {},
        "related": [
            "How-to Guidance"
        ]
    },
    {
        "name": "Permit or Document Process",
        "level": 4,
        "parent": "Public Administration",
        "en_category": "Permit",
        "description": "Administrative applications, forms, and approvals",
        "lang_categories": {},
        "related": [
            "Rules & Enforcement"
        ]
    },
    {
        "name": "Utility Service Access",
        "level": 4,
        "parent": "Public Services",
        "en_category": "Utility",
        "description": "Water, electricity, transport, or similar service delivery",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Legal Obligation",
        "level": 4,
        "parent": "Rights & Obligations",
        "en_category": "Obligation",
        "description": "What a person or institution must do under a rule",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Policy Reform Proposal",
        "level": 4,
        "parent": "Policy Proposal",
        "en_category": "Reform",
        "description": "A suggested change to laws, institutions, or public programs",
        "lang_categories": {},
        "related": [
            "Public Debate"
        ]
    },
    {
        "name": "Election Campaign Speech",
        "level": 4,
        "parent": "Campaign Messaging",
        "en_category": "Campaign speech",
        "description": "A candidate argues for support in public",
        "lang_categories": {},
        "related": [
            "Public Debate"
        ]
    },
    {
        "name": "Voting Instructions",
        "level": 4,
        "parent": "Voting Process",
        "en_category": "Voting",
        "description": "Telling citizens how, when, or where to vote",
        "lang_categories": {},
        "related": [
            "How-to Guidance"
        ]
    },
    {
        "name": "Parent Guidance",
        "level": 4,
        "parent": "Parent–Child Interaction",
        "en_category": "Parenting",
        "description": "A parent instructs, reassures, or corrects a child",
        "lang_categories": {},
        "related": [
            "Request"
        ]
    },
    {
        "name": "Family Planning Talk",
        "level": 4,
        "parent": "Family Discussion",
        "en_category": "Family",
        "description": "Family members organize meals, visits, or household decisions",
        "lang_categories": {},
        "related": [
            "Joint Planning"
        ]
    },
    {
        "name": "Friendly Check-in",
        "level": 4,
        "parent": "Casual Socializing",
        "en_category": "Friendship",
        "description": "Informal conversation about how someone is doing",
        "lang_categories": {},
        "related": [
            "Greeting Exchange"
        ]
    },
    {
        "name": "Emotional Reassurance",
        "level": 4,
        "parent": "Empathy & Support",
        "en_category": "Support",
        "description": "Comforting or encouraging another person",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Relationship Argument",
        "level": 4,
        "parent": "Relationship Tension",
        "en_category": "Relationship counseling",
        "description": "Personal conflict between partners",
        "lang_categories": {},
        "related": [
            "Verbal Dispute"
        ]
    },
    {
        "name": "Price Bargaining",
        "level": 4,
        "parent": "Bargaining",
        "en_category": "Price negotiation",
        "description": "Negotiating a lower price or different terms",
        "lang_categories": {},
        "related": [
            "Transactions"
        ]
    },
    {
        "name": "Project Agreement",
        "level": 4,
        "parent": "Agreement Formation",
        "en_category": "Agreement",
        "description": "People align on a plan, deal, or next steps",
        "lang_categories": {},
        "related": [
            "Joint Planning"
        ]
    },
    {
        "name": "Household Argument",
        "level": 4,
        "parent": "Verbal Dispute",
        "en_category": "Argument",
        "description": "A disagreement inside domestic or close relationships",
        "lang_categories": {},
        "related": [
            "Family Discussion"
        ]
    },
    {
        "name": "Public Debate",
        "level": 4,
        "parent": "Debate",
        "en_category": "Public debate",
        "description": "Competing political or social arguments in public",
        "lang_categories": {},
        "related": [
            "Election Campaign Speech"
        ]
    },
    {
        "name": "Grammar Explanation",
        "level": 4,
        "parent": "Sentence Structure",
        "en_category": "Grammar",
        "description": "Explaining how sentences are formed",
        "lang_categories": {},
        "related": [
            "How-to Guidance"
        ]
    },
    {
        "name": "Concept Clarification",
        "level": 4,
        "parent": "Meaning Relations",
        "en_category": "Concept",
        "description": "Explaining the meaning of a term or idea",
        "lang_categories": {},
        "related": [
            "Clarification Exchange"
        ]
    },
    {
        "name": "Request–Response Dialogue",
        "level": 4,
        "parent": "Dialogue Scene",
        "en_category": "Conversation",
        "description": "One speaker asks, another responds",
        "lang_categories": {},
        "related": [
            "Question–Answer"
        ]
    },
    {
        "name": "Greeting Exchange",
        "level": 4,
        "parent": "Small Talk",
        "en_category": "Greeting",
        "description": "Short social opening between speakers",
        "lang_categories": {},
        "related": [
            "Friendly Check-in"
        ]
    },
    {
        "name": "Step-by-Step Instruction",
        "level": 4,
        "parent": "How-to Guidance",
        "en_category": "Instruction",
        "description": "Ordered list of actions to complete a task",
        "lang_categories": {},
        "related": [
            "Task Assignment"
        ]
    },
    {
        "name": "Calendar Coordination",
        "level": 4,
        "parent": "Planning & Scheduling",
        "en_category": "Scheduling",
        "description": "Aligning times, dates, and availability",
        "lang_categories": {},
        "related": [
            "Trip Planning"
        ]
    },
    {
        "name": "Personal Story",
        "level": 4,
        "parent": "Event Narrative",
        "en_category": "Storytelling",
        "description": "A speaker recounts something they experienced",
        "lang_categories": {},
        "related": [
            "Emotion Expression"
        ]
    },
    {
        "name": "News Report",
        "level": 4,
        "parent": "Factual Report",
        "en_category": "News",
        "description": "A formal account of current events or public facts",
        "lang_categories": {},
        "related": [
            "Politics"
        ]
    },
    {
        "name": "Choice Recommendation",
        "level": 4,
        "parent": "Recommendation Dialogue",
        "en_category": "Recommendation",
        "description": "Suggesting an option and giving reasons for it",
        "lang_categories": {},
        "related": [
            "Product Choice"
        ]
    },
    {
        "name": "Clarifying Question",
        "level": 4,
        "parent": "Clarification Exchange",
        "en_category": "Question",
        "description": "A question used to reduce ambiguity or verify understanding",
        "lang_categories": {},
        "related": [
            "Concept Clarification"
        ]
    },
    {
        "name": "Lyric Expression",
        "level": 4,
        "parent": "Song",
        "en_category": "Lyrics",
        "description": "Words used in musical expression",
        "lang_categories": {},
        "related": [
            "Poetry"
        ]
    },
    {
        "name": "Dramatic Scene",
        "level": 4,
        "parent": "Stage Interaction",
        "en_category": "Drama",
        "description": "A performed exchange of action and speech",
        "lang_categories": {},
        "related": [
            "Dialogue Scene"
        ]
    },
    {
        "name": "Measurement",
        "level": 5,
        "parent": "Wave Function",
        "en_category": "Measurement",
        "description": "Observation or quantification of a state or variable",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Heat Transfer",
        "level": 5,
        "parent": "Entropy",
        "en_category": "Heat transfer",
        "description": "Movement of thermal energy between systems",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Genes",
        "level": 5,
        "parent": "DNA Inheritance",
        "en_category": "Gene",
        "description": "Units of hereditary information",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Adaptation",
        "level": 5,
        "parent": "Natural Selection",
        "en_category": "Adaptation",
        "description": "Trait adjustment through selective pressures",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Universe Expansion",
        "level": 5,
        "parent": "Big Bang Model",
        "en_category": "Expansion of the universe",
        "description": "Increasing scale of cosmic structure over time",
        "lang_categories": {},
        "related": [
            "Cosmology"
        ]
    },
    {
        "name": "Choose",
        "level": 4,
        "parent": "Product Choice",
        "en_category": "Choice",
        "description": "Selecting one option among alternatives",
        "lang_categories": {},
        "related": [
            "Recommend"
        ]
    },
    {
        "name": "Buy",
        "level": 5,
        "parent": "Grocery Purchase",
        "en_category": "Purchase",
        "description": "Acquire a good through payment",
        "lang_categories": {},
        "related": [
            "Pay"
        ]
    },
    {
        "name": "Pay",
        "level": 4,
        "parent": "Purchase & Payment",
        "en_category": "Payment",
        "description": "Transfer money to complete a transaction",
        "lang_categories": {},
        "related": [
            "Refund"
        ]
    },
    {
        "name": "Travel",
        "level": 4,
        "parent": "Trip Planning",
        "en_category": "Travel",
        "description": "Go from one place to another",
        "lang_categories": {},
        "related": [
            "Move"
        ]
    },
    {
        "name": "Move",
        "level": 4,
        "parent": "Commuting",
        "en_category": "Movement",
        "description": "Change location in physical space",
        "lang_categories": {},
        "related": [
            "Travel"
        ]
    },
    {
        "name": "Cook",
        "level": 4,
        "parent": "Meal Preparation",
        "en_category": "Cooking",
        "description": "Prepare food using ingredients and actions",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Repair",
        "level": 5,
        "parent": "Home Repair Request",
        "en_category": "Repair",
        "description": "Fix a broken or faulty object/system",
        "lang_categories": {},
        "related": [
            "Troubleshoot"
        ]
    },
    {
        "name": "Assign",
        "level": 5,
        "parent": "Task Assignment",
        "en_category": "Assignment",
        "description": "Give responsibility for a task",
        "lang_categories": {},
        "related": [
            "Plan"
        ]
    },
    {
        "name": "Report",
        "level": 5,
        "parent": "Status Update",
        "en_category": "Reporting",
        "description": "State progress, facts, or current condition",
        "lang_categories": {},
        "related": [
            "Explain"
        ]
    },
    {
        "name": "Refund",
        "level": 5,
        "parent": "Refund Processing",
        "en_category": "Refund",
        "description": "Return money after an error or cancellation",
        "lang_categories": {},
        "related": [
            "Pay"
        ]
    },
    {
        "name": "Complain",
        "level": 5,
        "parent": "Double Charge Complaint",
        "en_category": "Complaint",
        "description": "Report dissatisfaction or a problem requiring correction",
        "lang_categories": {},
        "related": [
            "Request"
        ]
    },
    {
        "name": "Troubleshoot",
        "level": 5,
        "parent": "Technical Setup Help",
        "en_category": "Troubleshooting",
        "description": "Identify and resolve the cause of a problem",
        "lang_categories": {},
        "related": [
            "Explain"
        ]
    },
    {
        "name": "Apply",
        "level": 5,
        "parent": "Permit or Document Process",
        "en_category": "Application",
        "description": "Submit a request or form for approval",
        "lang_categories": {},
        "related": [
            "Request"
        ]
    },
    {
        "name": "Comply",
        "level": 5,
        "parent": "Legal Obligation",
        "en_category": "Compliance",
        "description": "Act according to a law, rule, or requirement",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Vote",
        "level": 5,
        "parent": "Voting Instructions",
        "en_category": "Voting",
        "description": "Cast a ballot or express electoral choice",
        "lang_categories": {},
        "related": [
            "Choose"
        ]
    },
    {
        "name": "Campaign",
        "level": 5,
        "parent": "Election Campaign Speech",
        "en_category": "Political campaign",
        "description": "Seek support for a candidate or cause",
        "lang_categories": {},
        "related": [
            "Persuade"
        ]
    },
    {
        "name": "Reassure",
        "level": 5,
        "parent": "Emotional Reassurance",
        "en_category": "Reassurance",
        "description": "Reduce another person’s worry or distress",
        "lang_categories": {},
        "related": [
            "Comfort"
        ]
    },
    {
        "name": "Comfort",
        "level": 4,
        "parent": "Empathy & Support",
        "en_category": "Comfort",
        "description": "Provide emotional support or relief",
        "lang_categories": {},
        "related": [
            "Reassure"
        ]
    },
    {
        "name": "Argue",
        "level": 5,
        "parent": "Household Argument",
        "en_category": "Argument",
        "description": "State opposing positions in conflict",
        "lang_categories": {},
        "related": [
            "Criticize"
        ]
    },
    {
        "name": "Negotiate",
        "level": 5,
        "parent": "Price Bargaining",
        "en_category": "Negotiation",
        "description": "Seek agreement through back-and-forth terms",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Agree",
        "level": 5,
        "parent": "Project Agreement",
        "en_category": "Agreement",
        "description": "Accept terms or converge on a decision",
        "lang_categories": {},
        "related": [
            "Confirm"
        ]
    },
    {
        "name": "Plan",
        "level": 4,
        "parent": "Joint Planning",
        "en_category": "Planning",
        "description": "Organize future actions and responsibilities",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Ask",
        "level": 5,
        "parent": "Clarifying Question",
        "en_category": "Question",
        "description": "Seek information, confirmation, or action",
        "lang_categories": {},
        "related": [
            "Answer"
        ]
    },
    {
        "name": "Answer",
        "level": 5,
        "parent": "Request–Response Dialogue",
        "en_category": "Answer",
        "description": "Respond to a question or request",
        "lang_categories": {},
        "related": [
            "Ask"
        ]
    },
    {
        "name": "Greet",
        "level": 5,
        "parent": "Greeting Exchange",
        "en_category": "Greeting",
        "description": "Open a social interaction politely",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Explain",
        "level": 5,
        "parent": "Concept Clarification",
        "en_category": "Explanation",
        "description": "Make something understandable",
        "lang_categories": {},
        "related": [
            "Clarify"
        ]
    },
    {
        "name": "Clarify",
        "level": 5,
        "parent": "Concept Clarification",
        "en_category": "Clarification",
        "description": "Reduce ambiguity or misunderstanding",
        "lang_categories": {},
        "related": [
            "Explain"
        ]
    },
    {
        "name": "Instruct",
        "level": 5,
        "parent": "Step-by-Step Instruction",
        "en_category": "Instruction",
        "description": "Tell someone what to do in ordered steps",
        "lang_categories": {},
        "related": [
            "Guide"
        ]
    },
    {
        "name": "Guide",
        "level": 4,
        "parent": "How-to Guidance",
        "en_category": "Guidance",
        "description": "Lead another person through a process",
        "lang_categories": {},
        "related": [
            "Instruct"
        ]
    },
    {
        "name": "Recommend",
        "level": 5,
        "parent": "Choice Recommendation",
        "en_category": "Recommendation",
        "description": "Suggest an option as preferable",
        "lang_categories": {},
        "related": [
            "Persuade"
        ]
    },
    {
        "name": "Persuade",
        "level": 5,
        "parent": "Public Debate",
        "en_category": "Persuasion",
        "description": "Try to change belief or choice through reasons",
        "lang_categories": {},
        "related": [
            "Argue"
        ]
    },
    {
        "name": "Criticize",
        "level": 4,
        "parent": "Critique & Rebuttal",
        "en_category": "Criticism",
        "description": "Point out faults or oppose a claim",
        "lang_categories": {},
        "related": [
            "Rebut"
        ]
    },
    {
        "name": "Rebut",
        "level": 4,
        "parent": "Critique & Rebuttal",
        "en_category": "Rebuttal",
        "description": "Respond against an argument with counter-reasoning",
        "lang_categories": {},
        "related": [
            "Argue"
        ]
    },
    {
        "name": "Narrate",
        "level": 5,
        "parent": "Personal Story",
        "en_category": "Narration",
        "description": "Tell a sequence of events",
        "lang_categories": {},
        "related": [
            "Describe"
        ]
    },
    {
        "name": "Describe",
        "level": 5,
        "parent": "News Report",
        "en_category": "Description",
        "description": "State properties, events, or conditions",
        "lang_categories": {},
        "related": [
            "Report"
        ]
    },
    {
        "name": "Apologize",
        "level": 5,
        "parent": "Relationship Argument",
        "en_category": "Apology",
        "description": "Express regret and seek repair",
        "lang_categories": {},
        "related": []
    },
    {
        "name": "Confirm",
        "level": 4,
        "parent": "Agreement Formation",
        "en_category": "Confirmation",
        "description": "State that something is accepted, correct, or settled",
        "lang_categories": {},
        "related": [
            "Agree"
        ]
    },
    {
        "name": "Request",
        "level": 5,
        "parent": "Request–Response Dialogue",
        "en_category": "Request",
        "description": "Ask another person to do or provide something",
        "lang_categories": {},
        "related": []
    }
]
