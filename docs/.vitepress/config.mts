import { defineConfig } from "vitepress";
import { resolve } from "node:path";

const repo = "https://github.com/Sunwood-ai-labs/MatAnyone2-Gradio-Windows";

export default defineConfig({
  title: "MatAnyone",
  description: "Windows-friendly docs for the MatAnyone and MatAnyone 2 local runtime.",
  base: "/MatAnyone2-Gradio-Windows/",
  cleanUrls: true,
  vite: {
    publicDir: resolve(__dirname, "../../media"),
  },
  themeConfig: {
    search: {
      provider: "local",
    },
    socialLinks: [{ icon: "github", link: repo }],
  },
  locales: {
    root: {
      label: "English",
      lang: "en",
      themeConfig: {
        nav: [
          { text: "Guide", link: "/guide/getting-started" },
          { text: "Architecture", link: "/guide/architecture" },
          { text: "CI/CD", link: "/guide/ci-cd" },
          { text: "GitHub", link: repo },
          { text: "日本語", link: "/ja/" },
        ],
        sidebar: [
          {
            text: "Guide",
            items: [
              { text: "Getting Started", link: "/guide/getting-started" },
              { text: "Usage", link: "/guide/usage" },
              { text: "Performance", link: "/guide/performance" },
              { text: "Architecture", link: "/guide/architecture" },
              { text: "CI/CD", link: "/guide/ci-cd" },
              { text: "Troubleshooting", link: "/guide/troubleshooting" },
            ],
          },
        ],
      },
    },
    ja: {
      label: "日本語",
      lang: "ja",
      link: "/ja/",
      themeConfig: {
        nav: [
          { text: "ガイド", link: "/ja/guide/getting-started" },
          { text: "アーキテクチャ", link: "/ja/guide/architecture" },
          { text: "CI/CD", link: "/ja/guide/ci-cd" },
          { text: "GitHub", link: repo },
          { text: "English", link: "/" },
        ],
        sidebar: [
          {
            text: "ガイド",
            items: [
              { text: "セットアップ", link: "/ja/guide/getting-started" },
              { text: "使い方", link: "/ja/guide/usage" },
              { text: "パフォーマンス", link: "/ja/guide/performance" },
              { text: "アーキテクチャ", link: "/ja/guide/architecture" },
              { text: "CI/CD", link: "/ja/guide/ci-cd" },
              { text: "トラブルシューティング", link: "/ja/guide/troubleshooting" },
            ],
          },
        ],
      },
    },
  },
});
