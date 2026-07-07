import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const preserveMarkdownBasename = ({ entry }: { entry: string }) => entry.replace(/\.md$/, "");

const projects = defineCollection({
	loader: glob({ pattern: "**/*.md", base: "./src/content/projects", generateId: preserveMarkdownBasename }),
	schema: z.object({
		title: z.string(),
		cardTitle: z.string().optional(),
		zhTitle: z.string().optional(),
		description: z.string(),
		zhDescription: z.string().optional(),
		date: z.string(),
		zhDate: z.string().optional(),
		tags: z.array(z.string()).default([]),
		image: z.string().optional(),
		featured: z.boolean().default(true),
		order: z.number().default(999),
		github: z.string().url().optional(),
		document: z.string().optional(),
		publication: z.string().url().optional(),
		backHref: z.string().optional(),
	}),
});

const articles = defineCollection({
	loader: glob({ pattern: "**/*.md", base: "./src/content/articles", generateId: preserveMarkdownBasename }),
	schema: z.object({
		title: z.string(),
		date: z.string(),
		zhDate: z.string().optional(),
		description: z.string(),
		zhTitle: z.string().optional(),
		zhDescription: z.string().optional(),
		lang: z.string().default("en"),
	}),
});

export const collections = { projects, articles };
