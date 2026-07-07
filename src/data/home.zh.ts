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
		licenseLabel: "基于 MIT 许可协议",
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
					"我的学习从应用数学开始。在本科阶段，我建立了数学建模、优化和统计学方面的基础，也逐渐对机器学习产生兴趣，并开始自学编程。我意识到，数学不仅可以解释抽象问题，也可以用于构建能够解决现实问题的智能系统。",
			},
			{
				date: "2021.09 - 2023.06",
				stage: "统计学硕士与机器学习",
				organization: "卡尔加里大学",
				icon: { type: "image", src: "/img/UofCCoat.svg.png", alt: "卡尔加里大学 logo" },
				description:
					"2021 年我来到加拿大，在卡尔加里大学攻读统计学硕士。我的研究聚焦于改进基因组预测中机器学习模型的交叉验证方法，将统计理论与大规模计算结合起来。与此同时，我也参与了机器人操作中的强化学习研究，并积累了学术研究和报告展示经验。",
			},
			{
				date: "2023",
				stage: "机器学习开发者",
				organization: "Tech Start UCalgary",
				icon: { type: "image", src: "/img/tech-start-black.png", alt: "Tech Start logo" },
				description:
					"进入工业界之前，我加入 Tech Start，与跨学科团队合作开发早期技术项目。这是我第一次在快节奏环境中参与软件产品构建，也让我学习到如何与不同技术背景的人协作。",
			},
			{
				date: "2023",
				stage: "数据科学家",
				organization: "Cenozon Inc.",
				icon: { type: "image", src: "/img/cenozon-logo.png", alt: "Cenozon logo", wide: true },
				description:
					"在 Cenozon，我处理工业管道数据，并开发用于腐蚀预测和风险分析的机器学习模型。这是我第一次将数据科学应用到关键基础设施和大型工程数据中，也让我学会了如何结合领域知识解决实际问题。",
			},
			{
				date: "2023 - 2024",
				stage: "数据科学家",
				organization: "Intact Financial Corporation",
				icon: { type: "image", src: "/img/intact-logo.svg", alt: "Intact Insurance logo", wide: true },
				description:
					"在 Intact，我基于大规模客户服务数据开发自然语言处理和生成式 AI 解决方案。我搭建机器学习流程，实验大语言模型，并与工程团队协作，让模型更接近真实生产环境。",
			},
			{
				date: "2024 - 至今",
				stage: "电力分析师",
				organization: "BBA Engineering",
				icon: { type: "image", src: "/img/bba-logo.svg", alt: "BBA logo" },
				description:
					"我的工作逐渐转向能源分析，将机器学习、预测、优化和软件工程应用到电力系统问题中。相关项目包括负荷预测、电池储能优化、输电资产风险分析、OT/SCADA 网络安全自动化，以及面向电力公司的决策支持工具。",
			},
			{
				date: "今天",
				stage: "灵感驱动开发者",
				organization: "构建智能系统",
				icon: { type: "ion", name: "ion-network" },
				description:
					"现在，我关注如何将 AI、优化和数据科学应用到复杂现实系统中。无论是在能源、金融还是其他数据密集型领域，我都希望构建软件和智能决策工具，帮助人们解决困难问题。工作之外，我也持续探索 AI 辅助开发、新技术实验，以及把想法变成产品的软件项目。",
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
		lead: "我的项目涵盖数据分析、机器学习、统计建模和软件开发等方向。",
		readMoreLabel: "阅读全文",
	},
	about: {
		badge: "关于我",
		title: "阿奇（Archie Yanzhao Qian）",
		imageAlt: "Archibald Qian",
		paragraphs: [
			"你好，我是阿奇（Archie Yanzhao Qian）。我希望通过软件、数据与人工智能，让复杂系统变得更容易理解、更容易分析，也更容易做出决策。",
			"我主要关注人工智能、能源系统与软件工程的交叉领域。我对复杂系统充满兴趣，也乐于利用数据分析、预测模型和软件工程，将复杂的问题转化为更直观、更高效的工具。",
			"目前，我的工作和研究主要围绕电力系统、能源市场、预测与优化、AI 智能体，以及个人金融分析展开。这个网站记录了我的项目、研究和实验，也记录着我不断学习和构建的过程。",
		],
	},
	contact: {
		title: "联系我",
		lead: "如果您有任何问题，请联系我，我会尽快回复",
		text: "欢迎联系我交流项目、研究想法，或任何与人工智能、数据分析和软件工程相关的机会。",
		placeholders: { name: "姓名", email: "邮箱", subject: "标题", message: "内容", button: "发送" },
	},
};
