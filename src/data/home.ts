export type TimelineIcon =
	| {
			type: "image";
			src: string;
			alt: string;
			wide?: boolean;
	  }
	| {
			type: "ion";
			name: string;
	  };

export type TimelineItem = {
	date: string;
	stage: string;
	organization: string;
	icon: TimelineIcon;
	description?: string;
};

export type ProjectPreview = {
	image: string;
	date: string;
	title: string;
	description: string;
	href?: string;
	links?: Array<{
		type: "github" | "document" | "external";
		href: string;
		label: string;
	}>;
};

export type HomeContent = {
	title: string;
	lang?: string;
	languageHref: string;
	languageLabel: string;
	navItems: Array<{ href: string; label: string }>;
	hero: {
		lead: string;
		headingPrefix: string;
		highlight: string;
		headingSuffix: string;
		ctaLabel: string;
	};
	journeySection: {
		title: string;
		items: TimelineItem[];
	};
	articlesSection: {
		title: string;
		lead: string;
		readMoreLabel: string;
		viewAllLabel: string;
	};
	projectsSection: {
		title: string;
		lead: string;
		readMoreLabel: string;
		items: ProjectPreview[];
	};
	about: {
		badge: string;
		title: string;
		imageAlt: string;
		paragraphs: string[];
	};
	contact: {
		title: string;
		lead: string;
		text: string;
		placeholders: {
			name: string;
			email: string;
			subject: string;
			message: string;
			button: string;
		};
	};
};
