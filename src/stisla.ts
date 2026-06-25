type ArticleRecord = Record<string, string>;

interface ContactResponse {
	status: boolean;
	data: string;
}

interface JQueryCollection {
	[index: number]: HTMLElement;
	length: number;
	addClass(className: string): JQueryCollection;
	animate(properties: Record<string, number>): JQueryCollection;
	append(content: string): JQueryCollection;
	attr(name: string): string | undefined;
	click(handler: (this: HTMLElement, event: JQueryEvent) => false | void): JQueryCollection;
	complete?: unknown;
	css(properties: Record<string, string>): JQueryCollection;
	each(handler: (this: HTMLElement, index: number, element: HTMLElement) => void): JQueryCollection;
	easeScroll(): JQueryCollection;
	fadeIn(): JQueryCollection;
	fadeOut(callback?: () => void): JQueryCollection;
	offset(): { top: number } | undefined;
	on(eventName: string, selector: string, handler: (this: HTMLElement, event: JQueryEvent) => false | void): JQueryCollection;
	outerHeight(): number;
	parent(): JQueryCollection;
	prepend(content: string): JQueryCollection;
	remove(): JQueryCollection;
	removeClass(className: string): JQueryCollection;
	reset?: () => void;
	scroll(handler: (this: Window, event: JQueryEvent) => void): JQueryCollection;
	scrollTop(): number;
	serialize(): string;
	submit(handler: (this: HTMLFormElement, event: JQueryEvent) => false | void): JQueryCollection;
}

interface JQueryEvent {
	preventDefault(): void;
}

interface JQueryAjaxSettings<TResponse> {
	url: string;
	type?: string;
	data?: string;
	dataType: "json";
	beforeSend?: () => void;
	complete?: () => void;
	success?: (data: TResponse) => void;
}

interface JQueryStatic {
	(selector: string | HTMLElement | Document | Window): JQueryCollection;
	(callback: () => void): void;
	ajax<TResponse>(settings: JQueryAjaxSettings<TResponse>): void;
}

interface SweetAlert {
	(title: string, message: string, type: "success" | "error"): void;
}

declare const $: JQueryStatic;
declare const swal: SweetAlert;

const selectors = {
	articleBackButton: ".article-back .btn",
	background: "[data-bg]",
	body: "body",
	collapseToggle: "[data-toggle=collapse][data-target]",
	contactForm: "#contact-form",
	mainNavbar: ".main-navbar",
	mainLoading: ".main-loading",
	readToggle: "[data-toggle=read]",
	section: "section",
	smoothLink: ".smooth-link",
} as const;

const articleTemplate = `
<div class="article-read">
	<div class="article-read-inner">
		<div class="article-back">
			<a class="btn btn-outline-primary"><i class="ion ion-chevron-left"></i> Back</a>
		</div>
		<h1 class="article-title">{title}</h1>
		<div class="article-metas">
			<div class="meta">{date}</div>
			<div class="meta">{category}</div>
			<div class="meta">{author}</div>
		</div>
		<figure class="article-picture"><img src="{picture}"></figure>
		<div class="article-content">{content}</div>
	</div>
</div>`;

const loading = {
	show(): void {
		$(selectors.body).append("<div class='main-loading'></div>");
	},

	hide(): void {
		$(selectors.mainLoading).remove();
	},
};

function setupSmoothPageScroll(): void {
	$(selectors.body).easeScroll();
}

function setupBackgroundImages(): void {
	$(selectors.background).each(function () {
		const $element = $(this);
		const backgroundUrl = $element.attr("data-bg");

		if (!backgroundUrl) {
			return;
		}

		$element.css({
			backgroundImage: `url(${backgroundUrl})`,
			backgroundPosition: "center",
			backgroundAttachment: "fixed",
			backgroundSize: "center",
		});
		$element.prepend("<div class='overlay'></div>");
	});
}

function isHashLink(href: string | undefined): href is string {
	return Boolean(href && href.startsWith("#") && href.length > 1);
}

function setupAnchorScrolling(): void {
	$(selectors.smoothLink).click(function () {
		const href = $(this).attr("href");

		if (!isHashLink(href)) {
			return;
		}

		const $target = $(href);
		const targetOffset = $target.offset();
		const navbarHeight = $(selectors.mainNavbar).outerHeight();

		if (!targetOffset) {
			return;
		}

		$("html, body").animate({
			scrollTop: targetOffset.top - (navbarHeight - 1),
		});

		return false;
	});
}

function setupNavbarScrollState(): void {
	$(window).scroll(function () {
		const $window = $(this);
		const heroHeight = $(".hero").outerHeight();
		const navbarHeight = $(selectors.mainNavbar).outerHeight();

		if ($window.scrollTop() > heroHeight / 10) {
			$(selectors.mainNavbar).addClass("bg-dark");
		} else {
			$(selectors.mainNavbar).removeClass("bg-dark");
		}

		$(selectors.section).each(function () {
			const sectionId = $(this).attr("id");
			const sectionOffset = $(this).offset();

			if (!sectionId || !sectionOffset) {
				return;
			}

			if ($window.scrollTop() >= sectionOffset.top - navbarHeight) {
				$(selectors.smoothLink).parent().removeClass("active");
				$(`${selectors.smoothLink}[href="#${sectionId}"]`).parent().addClass("active");
			}
		});
	});
}

function setupCollapseToggles(): void {
	document.querySelectorAll<HTMLElement>(selectors.collapseToggle).forEach((toggle) => {
		const targetSelector = toggle.getAttribute("data-target");

		if (!targetSelector) {
			return;
		}

		const targetElement = document.querySelector(targetSelector);

		if (!targetElement) {
			return;
		}

		toggle.addEventListener("click", (event) => {
			const isExpanded = targetElement.classList.toggle("show");
			toggle.setAttribute("aria-expanded", String(isExpanded));
			event.preventDefault();
		});
	});
}

function renderArticle(template: string, data: ArticleRecord): string {
	return template.replace(/{([a-zA-Z0-9]+)}/g, (_placeholder, key: string) => data[key] ?? "");
}

function closeArticle(): false {
	$(".article-read").fadeOut(function () {
		$(".article-read").remove();
		$(selectors.body).css({
			overflow: "auto",
		});
	});

	return false;
}

function setupArticleReader(): void {
	$(selectors.readToggle).click(function () {
		$(selectors.body).css({
			overflow: "hidden",
		});

		$.ajax<ArticleRecord>({
			url: "mock/article.json",
			dataType: "json",
			beforeSend: loading.show,
			complete: loading.hide,
			success(data) {
				$(selectors.body).prepend(renderArticle(articleTemplate, data));
				$(".article-read").fadeIn();
				$(document).on("click", selectors.articleBackButton, closeArticle);
			},
		});

		return false;
	});
}

function setupContactForm(): void {
	$(selectors.contactForm).submit(function () {
		const $form = $(this);

		$.ajax<ContactResponse>({
			url: "server/send.php",
			type: "post",
			data: $form.serialize(),
			dataType: "json",
			beforeSend: loading.show,
			complete: loading.hide,
			success(data) {
				if (data.status === true) {
					swal("Success", data.data, "success");
					($form[0] as HTMLFormElement).reset();
				} else {
					swal("Failed", data.data, "error");
				}
			},
		});

		return false;
	});
}

function initializeSite(): void {
	setupSmoothPageScroll();
	setupBackgroundImages();
	setupAnchorScrolling();
	setupNavbarScrollState();
	setupCollapseToggles();
	setupArticleReader();
	setupContactForm();
}

$(initializeSite);
