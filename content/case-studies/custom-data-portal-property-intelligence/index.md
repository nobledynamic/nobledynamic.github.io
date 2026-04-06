---
title: "Custom Data Portal for Global Property Intelligence"
summary: "Self-service data portal replacing legacy Excel reports with interactive querying, custom scenarios, and branded exports."
industry: "Property Data / Financial Services"
date: 2026-04-06
draft: false
---

## Property Data / Financial Services

A property data provider needed to modernise how their clients accessed data. The existing system relied on manually generated Excel files built with legacy tooling, copied as flat files to a website. Clients had no ability to explore, filter, or customise the data they needed and could only download pre-built reports. The company wanted to give clients a self-service experience: the ability to select datasets, choose variables, apply custom scenarios, and export branded, presentation-ready outputs.

We were brought in to design and build the solution from the ground up. The project started with a proof of concept to demonstrate viability to the company's partners and directors, and then progressed into a full production build.

The first phase involved user interface design workshops with stakeholders to understand how analysts and clients actually worked with the data, what they needed to filter on, how they thought about time series, and what a useful export looked like. This informed the design of a web-based data portal that allowed clients to query across markets, regions, and sectors, apply custom variables such as different interest rates, yields, and currencies to forecast data, and save queries for repeat use.

We designed the platform around a microservices architecture, with each core function: authentication, data API, user preferences, and export, built as independent, containerised services. This gave the client flexibility to deploy on-premise, in the cloud, or in a hybrid setup, and meant their in-house team could evolve individual services without risk to the wider system. The front end was built in Svelte, the API layer in .NET, and the whole platform orchestrated with Kubernetes.

A significant part of the value was in the export functionality. Clients expected branded Excel workbooks with the company's logo, disclaimers, source information, and consistent formatting, matching the quality of the manually produced reports they were used to, but generated on demand from live data. CSV exports were also supported for clients who wanted raw data for their own analysis.

For clients who wanted to integrate the data directly into their own systems, we also built a client-facing API. This gave programmatic access to the same datasets available through the web portal, allowing customers to pull data into their own tools, models, and workflows without needing to go through the UI.

Beyond the build itself, we worked closely with the in-house technical team on technology selection, architecture decisions, and training, helping them adopt modern practices around containerisation and microservices so they could maintain and extend the platform independently.
