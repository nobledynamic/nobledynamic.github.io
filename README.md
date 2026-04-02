# Noble Dynamic Website

Corporate website and blog for [Noble Dynamic](https://nobledynamic.com), a data services consultancy. Built with [Hugo](https://gohugo.io/) using the [Blowfish](https://github.com/nunocoracao/blowfish) theme and deployed to GitHub Pages.

## Tech Stack

- **Hugo** (extended) — static site generator
- **Blowfish** — Hugo theme with TailwindCSS
- **GitHub Actions** — automated build and deployment

## Getting Started

### Dev Container (recommended)

Open the project in VS Code with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension. The container automatically installs Hugo, initialises submodules, and forwards port 1313.

Then start the dev server:

```bash
hugo server
```

### Manual Setup

```bash
# Clone with submodules (fetches the Blowfish theme)
git clone --recurse-submodules https://github.com/nobledynamic/nobledynamic.github.io.git
cd nobledynamic.github.io

# Start the dev server
hugo server
```

> **Note:** Requires [Hugo extended](https://gohugo.io/installation/) to be installed.

The site will be available at `http://localhost:1313`.

## Project Structure

```
config/         # Hugo configuration (site settings, menus, theme params)
content/        # Markdown content (blog posts, legal pages)
layouts/        # Custom HTML templates and partials
static/         # Static assets (images, fonts, favicons)
data/           # Structured data (author profiles)
themes/         # Blowfish theme (git submodule)
```

## Deployment

Pushes to `main` automatically build and deploy the site via GitHub Actions. The workflow runs `hugo --minify` and publishes the output to GitHub Pages.
