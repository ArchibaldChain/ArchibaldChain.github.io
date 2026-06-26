import type { HomeContent } from "./home";

export const homeContent: HomeContent = {
	title: "I'm Archibald",
	languageHref: "/Chinese.html",
	languageLabel: "中文",
	navItems: [
		{ href: "#hero", label: "Home" },
		{ href: "#Experience", label: "Experience" },
		{ href: "#articles", label: "Articles" },
		{ href: "#project", label: "Projects" },
		{ href: "#aboutme", label: "About Me" },
		{ href: "#contact", label: "Contact Me" },
	],
	hero: {
		lead: "We should consider every day lost on which we have not danced at least once. --Nietzsche",
		headingPrefix: "I'm ",
		highlight: "Archibald",
		headingSuffix: ". Welcome to my website.",
		ctaLabel: "About Me",
	},
	journeySection: {
		title: "My Journey",
		items: [
			{
				date: "2017-2021",
				stage: "Applied Mathematics",
				organization: "Southern University of Science and Technology (China)",
				icon: { type: "image", src: "/img/sustech.png", alt: "Southern University of Science and Technology logo" },
				description:
					"My journey began with a degree in Applied Mathematics, where I developed a strong foundation in mathematical modeling, optimization, and statistics. During this time, I became fascinated by machine learning and started teaching myself programming, realizing that mathematics could be used to build intelligent systems capable of solving real-world problems.",
			},
			{
				date: "2021-2023",
				stage: "MSc Statistics & Machine Learning",
				organization: "University of Calgary",
				icon: { type: "image", src: "/img/UofCCoat.svg.png", alt: "University of Calgary logo" },
				description:
					"I moved to Canada in 2021 to pursue an MSc in Statistics at the University of Calgary. My research focused on improving cross-validation methods for machine learning models in genomic prediction, combining statistical theory with large-scale computation. Alongside my thesis, I also worked on reinforcement learning for robotic manipulation and participated in academic research and presentations.",
			},
			{
				date: "2023",
				stage: "ML Developer",
				organization: "Tech Start UCalgary",
				icon: { type: "image", src: "/img/tech-start-black.png", alt: "Tech Start logo" },
				description:
					"Before entering industry, I joined Tech Start to collaborate with multidisciplinary teams on early-stage technology projects. It was my first experience building software products in a fast-paced environment and working closely with people from different technical backgrounds.",
			},
			{
				date: "2023",
				stage: "Data Scientist",
				organization: "Cenozon Inc.",
				icon: { type: "image", src: "/img/cenozon-logo.png", alt: "Cenozon logo", wide: true },
				description:
					"At Cenozon, I worked with industrial pipeline data, developing machine learning models for corrosion prediction and risk analysis. This was my first opportunity to apply data science to critical infrastructure and large engineering datasets.",
			},
			{
				date: "2023-2024",
				stage: "Data Scientist",
				organization: "Intact Financial Corporation",
				icon: { type: "image", src: "/img/intact-logo.svg", alt: "Intact Insurance logo", wide: true },
				description:
					"At Intact, I developed NLP and generative AI solutions using large-scale customer service data. I built machine learning pipelines, experimented with LLMs, and collaborated with engineering teams to bring models closer to production environments.",
			},
			{
				date: "2024-Present",
				stage: "Power Analytics",
				organization: "BBA Engineering",
				icon: { type: "image", src: "/img/bba-logo.svg", alt: "BBA logo" },
				description:
					"My work has evolved toward energy analytics, where I apply machine learning, forecasting, optimization, and software engineering to power system challenges. My projects include electricity load forecasting, battery energy storage optimization, transmission asset risk analysis, OT/SCADA cybersecurity automation, and decision-support tools for electric utilities.",
			},
			{
				date: "Today",
				stage: "Vibe Developer",
				organization: "Building Intelligent Systems",
				icon: { type: "ion", name: "ion-network" },
				description:
					"Today, I'm interested in applying AI, optimization, and data science to complex real-world systems. Whether in energy, finance, or other data-intensive industries, I enjoy building software and intelligent decision-support tools that help people solve challenging problems. Outside of work, I continue exploring AI-assisted development, experimenting with new technologies, and building software products that turn ideas into reality.",
			},
		],
	},
	articlesSection: {
		title: "Articles",
		lead: "Notes, essays, and updates from what I am learning and building.",
		readMoreLabel: "Read More",
		viewAllLabel: "View All Articles",
	},
	projectsSection: {
		title: "My Projects",
		lead: "The projects I have done cover data analysis, machine learning, computer vision, etc.",
		readMoreLabel: "Read More",
		items: [
			{
				image: "/projects/CVc/bioinformatics.jpg",
				date: "Oct. 2021 - Present",
				title: "Cross-validation Correction For Machine Learing in Genomics Datasets",
				description:
					"In Genomics Dataset, every indivdual has very similar gene except a few variants which causes machine learning methods are easily over-fitting. Therefore CV which is used for estimating test error is underestimated.",
				href: "/projects/project-CVc.html",
				links: [
					{
						type: "github",
						href: "https://github.com/ArchibaldChain/CVc_in_bio_informatics",
						label: "View GitHub repository",
					},
				],
			},
			{
				image: "/projects/Internet Speed/internet.jpg",
				date: "May 2022",
				title: "Statistical Analysis of Ookla Internet Speeds for Rural/Urban Canadian Communities",
				description:
					"We visualized, processed, and analyzed the Internet Speed dataset provided by Ookla company. We used logistic regression to predict future internet speed situation and gave some suggestions according to it.",
				links: [
					{
						type: "github",
						href: "https://github.com/HH197/Case-Study-Competition",
						label: "View GitHub repository",
					},
					{
						type: "document",
						href: "/projects/Internet Speed/Interenet Speedposter.pdf",
						label: "Open project poster PDF",
					},
				],
			},
		],
	},
	about: {
		badge: "ABOUT ME",
		title: "Archibald - A Machine Learning Enthusiast",
		imageAlt: "Youzhang",
		paragraphs: [
			"I am Yanzhao Qian. You can also call me Archibald. I was born and raised in China, and finished my bachelor's degree in applied mathematics there. In order to persuade the master's degree at U of C I relocated to Calgary in 2021.",
			"My interest in machine learning drives me to study further in the field of statistics and to teach myself coding. I mainly focus on building a stronger artificial intelligence that can assists people, and analyzing historical datasets to make future predictions.",
			"In my free time, I enjoy trying and learning new things, watching movies, playing video games and boxing. Also, I've just started snowboarding recently.",
		],
	},
	contact: {
		title: "Contact Me",
		lead: "Send me your inquiries, and I will reply as soon as possible",
		text: "You can send me something like a question or project. Please contact me if you think I will be a good fit for the position.",
		placeholders: { name: "Name", email: "Email", subject: "Subject", message: "Message", button: "Send Message" },
	},
};
