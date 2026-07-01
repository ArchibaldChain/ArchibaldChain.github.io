import type { HomeContent } from "./home";

export const homeContent: HomeContent = {
	title: "我是阿奇",
	lang: "zh-CN",
	languageHref: "/",
	languageLabel: "English",
	navItems: [
		{ href: "#hero", label: "主页" },
		{ href: "#Experience", label: "经历" },
		{ href: "#articles", label: "文章" },
		{ href: "#project", label: "项目" },
		{ href: "#aboutme", label: "关于我" },
		{ href: "#contact", label: "联系我" },
	],
	hero: {
		lead: "每一个不曾起舞的日子都是对生命的辜负。——尼采",
		headingPrefix: "我是",
		highlight: "阿奇",
		headingSuffix: "，欢迎来到我的网站",
		ctaLabel: "关于我",
	},
	journeySection: {
		title: "我的经历",
		items: [
			{
				date: "2017.09 - 2021.06",
				stage: "数学与应用数学",
				organization: "南方科技大学",
				icon: { type: "image", src: "/img/sustech.png", alt: "南方科技大学 logo" },
				description:
					"我对理解机器学习的兴趣驱使我学习应用数学。这让我可以进一步研究算法。在学习数学的过程中，我学会了如何进行严格的证明，并且了解了现代数学是如何建构起来的。",
			},
			{
				date: "2021.09 - 2023.06",
				stage: "统计学硕士与机器学习",
				organization: "卡尔加里大学",
				icon: { type: "image", src: "/img/UofCCoat.svg.png", alt: "卡尔加里大学 logo" },
				description:
					"在这个阶段，我学习了广义线性模型、计算统计、动手深度学习等，为我的数据分析打下了坚实的基础。我的研究重点是生物信息学中的交叉验证问题。",
			},
			{
				date: "2023",
				stage: "机器学习开发者",
				organization: "Tech Start UCalgary",
				icon: { type: "image", src: "/img/tech-start-black.png", alt: "Tech Start logo" },
			},
			{
				date: "2023",
				stage: "数据科学家",
				organization: "Cenozon Inc.",
				icon: { type: "image", src: "/img/cenozon-logo.png", alt: "Cenozon logo", wide: true },
			},
			{
				date: "2023 - 2024",
				stage: "数据科学家",
				organization: "Intact Financial Corporation",
				icon: { type: "image", src: "/img/intact-logo.svg", alt: "Intact Insurance logo", wide: true },
			},
			{
				date: "2024 - 至今",
				stage: "电力分析师",
				organization: "BBA Engineering",
				icon: { type: "image", src: "/img/bba-logo.svg", alt: "BBA logo" },
			},
			{
				date: "Today",
				stage: "Vibe Developer",
				organization: "构建智能系统",
				icon: { type: "ion", name: "ion-network" },
			},
		],
	},
	articlesSection: {
		title: "文章",
		lead: "记录我正在学习、思考和构建的内容。",
		readMoreLabel: "阅读全文",
		viewAllLabel: "查看全部文章",
	},
	projectsSection: {
		title: "我的项目",
		lead: "我做过的项目涵盖与数据分析，机器学习，计算机视觉等。",
		readMoreLabel: "阅读全文",
		items: [
			{
				image: "/projects/CVc/cvc-cover.png",
				date: "2023-09",
				title: "Cross-validation Correction For Machine Learing in Genomics Datasets",
				description:
					"In Genomics Dataset, every indivdual has very similar gene except a few variants which causes machine learning methods are easily over-fitting. Therefore CV which is used for estimating test error is underestimated.",
				href: "/projects/project-CVc.html",
				links: [
					{
						type: "github",
						href: "https://github.com/ArchibaldChain/CVc_in_bio_informatics",
						label: "查看 GitHub 仓库",
					},
					{
						type: "external",
						href: "https://www.tandfonline.com/doi/full/10.1080/02664763.2026.2646570",
						label: "查看发表文章",
					},
				],
			},
			{
				image: "/projects/Internet Speed/canada-internet-cover.png",
				date: "May 2022",
				title: "Statistical Analysis of Ookla Internet Speeds for Rural/Urban Canadian Communities",
				description:
					"We visualized, processed, and analyzed the Internet Speed dataset provided by Ookla company. We used logistic regression to predict future internet speed situation and gave some suggestions according to it.",
				links: [
					{
						type: "github",
						href: "https://github.com/HH197/Case-Study-Competition",
						label: "查看 GitHub 仓库",
					},
					{
						type: "document",
						href: "/projects/Internet Speed/Interenet Speedposter.pdf",
						label: "打开项目海报 PDF",
					},
				],
			},
		],
	},
	about: {
		badge: "关于我",
		title: "阿奇 - 机器学习爱好者",
		imageAlt: "Youzhang",
		paragraphs: [
			"I am Yanzhao Qian. You can also call me Archibald. I was born and raised in China, and finished my bachelor's degree in applied mathematics there.",
			"My interest in machine learning drives me to study further in the field of statistics and to teach myself coding.",
			"In my free time, I enjoy trying and learning new things, watching movies, playing video games and boxing. Also, I've just started snowboarding recently.",
		],
	},
	contact: {
		title: "联系我",
		lead: "如果您有任何问题，请联系我，我会尽快回复",
		text: "您可以向我发送您的问题或者项目。如果您有认为我适合的职位，也请与我联系。",
		placeholders: { name: "姓名", email: "邮箱", subject: "标题", message: "内容", button: "发送" },
	},
};
