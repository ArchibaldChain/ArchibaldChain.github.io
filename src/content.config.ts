import { defineCollection, z } from "astro:content";

const projects = defineCollection({
	schema: z.object({
		title: z.string(),
		date: z.string(),
		tags: z.array(z.string()).default([]),
		image: z.string().optional(),
		github: z.string().url().optional(),
		document: z.string().optional(),
		backHref: z.string().optional(),
	}),
});

const articles = defineCollection({
	schema: z.object({
		title: z.string(),
		date: z.string(),
		description: z.string(),
		lang: z.string().default("en"),
	}),
});

export const collections = { projects, articles };
