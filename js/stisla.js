"use strict";
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
};
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
    show() {
        $(selectors.body).append("<div class='main-loading'></div>");
    },
    hide() {
        $(selectors.mainLoading).remove();
    },
};
function setupSmoothPageScroll() {
    $(selectors.body).easeScroll();
}
function setupBackgroundImages() {
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
function isHashLink(href) {
    return Boolean(href && href.startsWith("#") && href.length > 1);
}
function setupAnchorScrolling() {
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
function setupNavbarScrollState() {
    $(window).scroll(function () {
        const $window = $(this);
        const heroHeight = $(".hero").outerHeight();
        const navbarHeight = $(selectors.mainNavbar).outerHeight();
        if ($window.scrollTop() > heroHeight / 10) {
            $(selectors.mainNavbar).addClass("bg-dark");
        }
        else {
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
function setupCollapseToggles() {
    document.querySelectorAll(selectors.collapseToggle).forEach((toggle) => {
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
function renderArticle(template, data) {
    return template.replace(/{([a-zA-Z0-9]+)}/g, (_placeholder, key) => { var _a; return (_a = data[key]) !== null && _a !== void 0 ? _a : ""; });
}
function closeArticle() {
    $(".article-read").fadeOut(function () {
        $(".article-read").remove();
        $(selectors.body).css({
            overflow: "auto",
        });
    });
    return false;
}
function setupArticleReader() {
    $(selectors.readToggle).click(function () {
        $(selectors.body).css({
            overflow: "hidden",
        });
        $.ajax({
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
function setupContactForm() {
    $(selectors.contactForm).submit(function () {
        const $form = $(this);
        $.ajax({
            url: "server/send.php",
            type: "post",
            data: $form.serialize(),
            dataType: "json",
            beforeSend: loading.show,
            complete: loading.hide,
            success(data) {
                if (data.status === true) {
                    swal("Success", data.data, "success");
                    $form[0].reset();
                }
                else {
                    swal("Failed", data.data, "error");
                }
            },
        });
        return false;
    });
}
function initializeSite() {
    setupSmoothPageScroll();
    setupBackgroundImages();
    setupAnchorScrolling();
    setupNavbarScrollState();
    setupCollapseToggles();
    setupArticleReader();
    setupContactForm();
}
$(initializeSite);
