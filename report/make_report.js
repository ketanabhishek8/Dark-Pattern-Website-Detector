// Builds the project report from the numbers in figures/. See report/README.md.
// Usage: node report/make_report.js <output.docx> [--with-ids]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, LevelFormat, ExternalHyperlink, VerticalAlign,
} = require("docx");

const FIG = path.join(__dirname, "..", "figures");
const OUT = process.argv[2];
const WITH_IDS = process.argv.includes("--with-ids");
const MARGIN = 1020; // 1.8 cm
const W = 11906 - 2 * MARGIN; // content width in DXA

// Classic report styling: Calibri, navy headings and table headers, thin grey rules
const NAVY = "1F3864", GREY = "555555", RULE = "BFBFBF", CODE_BG = "F2F4F7";
const FONT = "Calibri";

const load = (f) => JSON.parse(fs.readFileSync(path.join(FIG, f), "utf8"));
const NB = load("results.json");                          // notebook: baselines, random split
const FINAL = load("eval_final_grouped.json");            // final model, split by website
const RANDOM = load("eval_random_split.json");            // final config, random split (leakage check)
const NOMINE = load("eval_no_mining.json");               // final config without mined lines
const RP_FIRST = load("real_pages_eval_first_model.json");
const RP_NOMINE = load("real_pages_eval_no_mining.json");
const RP = load("real_pages_eval.json");
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

// ---------- building blocks ----------
function runs(text, opts = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) => part.startsWith("**")
    ? new TextRun({ text: part.slice(2, -2), bold: true, ...opts })
    : new TextRun({ text: part, ...opts }));
}
const p = (text, extra = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 90 }, ...extra });
const h1 = (num, title) => new Paragraph({ text: num ? `${num}. ${title}` : title, heading: HeadingLevel.HEADING_1, keepNext: true });
const bullet = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 } });
const caption = (label, text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 40, after: 140 },
  children: runs(`${label.replace(/\.$/, ":")} ${text}`, { italics: true, size: 17, color: GREY }),
});
const image = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [
  new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)), transformation: { width: w, height: h } })] });
const line = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const grid = { top: line, bottom: line, left: line, right: line };

function cell(text, width, { head = false, bold = false, align = AlignmentType.CENTER } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: grid, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { type: ShadingType.CLEAR, fill: NAVY, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [new Paragraph({ alignment: head ? AlignmentType.CENTER : align, keepNext: true,
      children: [new TextRun({ text, bold: head || bold, color: head ? "FFFFFF" : undefined, size: 18 })] })],
  });
}

// A data table: navy header row, grey grid, an optional bold row
function table(header, rows, widths, { highlight = -1, leftCols = 1 } = {}) {
  const align = (i) => (i < leftCols ? AlignmentType.LEFT : AlignmentType.CENTER);
  return new Table({
    width: { size: widths.reduce((a, b) => a + b), type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: header.map((t, i) => cell(t, widths[i], { head: true })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) =>
        cell(t, widths[i], { bold: ri === highlight, align: align(i) })) })),
    ],
  });
}

// ---------- title and abstract ----------
const authors = WITH_IDS
  ? "Abhishek Joshi (16014124023)  &  Shaurya Ghorpade (16014124017)"
  : "Abhishek Joshi  &  Shaurya Ghorpade";
const title = [
  new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: [new TextRun("Dark Pattern Detector")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Detecting Manipulative Text on Indian E-Commerce Websites using NLP", size: 25, color: "404040" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 }, children: runs(`Submitted by: ${authors}`) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: runs("Course: Artificial Intelligence   |   Date: 30 September 2026") }),
];

const abstract = new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, indent: { left: 400, right: 400 },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 }, bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 6 } },
  children: runs(`**Abstract.** Dark patterns are pieces of interface text designed to pressure shoppers, such as fake countdown timers and false stock warnings. We fine-tuned DistilBERT to label each line of a product page as either not a dark pattern or one of five dark pattern types. On a benchmark of 2,356 e-commerce texts, split so that the test set contains only websites the model never saw, it reaches ${pct(FINAL.original_test.f1)} F1 and names the type correctly ${pct(FINAL.category_on_dark_test.accuracy)} of the time. Because the benchmark comes from Western sites, we also evaluated the whole system on hand-labelled Amazon.in, Snapdeal and Indian brand-store pages. Better page parsing, a calibrated decision threshold and retraining on mistakes mined from real pages cut false alarms from 42.8 to 0 per page on held-out test pages, and on a live Amazon.in phone page flags fell from 92 to 2, both real dark patterns. The model powers a web dashboard that scans any product page from its link.`, { size: 19 }),
});

// ---------- tables ----------
const typesTable = table(["Type", "What it does", "Example from the dataset", "Texts"], [
  ["Scarcity", "Claims stock is running out or demand is high", "\"Only 2 left in stock!\"", "418"],
  ["Urgency", "Puts a real or fake deadline on the decision", "\"Deal ends in 04:59\"", "210"],
  ["Social Proof", "Uses other shoppers' activity as pressure", "\"27 people bought this in the last hour\"", "312"],
  ["Misdirection", "Guilt-trips or steers the choice", "\"No thanks, I don't like saving money\"", "195"],
  ["Other", "Obstruction, sneaking and forced action (merged: too few examples)", "", "43"],
  ["Not a dark pattern", "Ordinary shop text", "\"Write a review\"", "1,178"],
], [1700, 3900, 3166, 1100], { leftCols: 3 });

const nbRow = (label, split, key) => [label, split, pct(NB[key].accuracy), pct(NB[key].precision), pct(NB[key].recall), pct(NB[key].f1)];
const o = FINAL.original_test;
const benchmarkTable = table(["Model", "Test split", "Accuracy", "Precision", "Recall", "F1"], [
  nbRow("TF-IDF + Logistic Regression", "Random", "TF-IDF + Logistic Regression"),
  nbRow("TF-IDF + Linear SVM", "Random", "TF-IDF + Linear SVM"),
  nbRow("DistilBERT, 2 classes (first model)", "Random", "DistilBERT (fine-tuned)"),
  ["DistilBERT, 6 classes (final)", "By website", pct(o.accuracy), pct(o.precision), pct(o.recall), pct(o.f1)],
], [3330, 1440, 1274, 1274, 1274, 1274], { highlight: 3, leftCols: 2 });

const stage = (r, total) => [r.f1.toFixed(2), `${Math.round(r.recall * total)} of ${total}`, r.false_alarms_per_page.toFixed(1)];
const shipped = `new extractor, shipped threshold ${RP.shipped_threshold}`;
const realRows = [
  ["First model, original page parser, threshold 0.5", RP_FIRST, "old extractor, threshold 0.5"],
  ["+ new page parser", RP_FIRST, "new extractor, threshold 0.5"],
  ["+ threshold 0.97", RP_FIRST, "new extractor, threshold 0.97"],
  ["6-class model, no mined data", RP_NOMINE, "new extractor, threshold 0.97"],
  ["6-class model + mined data (final)", RP, shipped],
].map(([label, run, key]) => [label, ...stage(run.test[key], 9), ...stage(run.unseen[key], 3)]);
const realTable = table(
  ["Pipeline", "Test F1", "Test found", "Test false alarms/page", "Unseen F1", "Unseen found", "Unseen false alarms/page"],
  realRows, [2986, 1000, 1080, 1400, 1000, 1080, 1320], { highlight: 4 });

const code = (text, last = false) => new Paragraph({
  keepNext: !last, spacing: { after: last ? 120 : 0 }, indent: { left: 360 },
  shading: { type: ShadingType.CLEAR, fill: CODE_BG, color: "auto" },
  children: [new TextRun({ text, font: "Courier New", size: 17 })],
});

// ---------- document ----------
const T = FINAL.category_on_dark_test;
const children = [
  ...title,
  abstract,

  h1("1", "Introduction and Problem Statement"),
  p("**Dark patterns** are interface designs that push people into decisions they did not intend to make. On shopping sites they are often just a line of text: a countdown (\"Deal ends in 04:59\"), a stock warning (\"Only 2 left!\"), a crowd claim (\"27 people bought this in the last hour\") or a guilt-trip opt-out (\"No thanks, I don't like saving money\"). A crawl of 11,000 shopping sites found 1,818 such instances (Mathur et al., 2019), and India's Central Consumer Protection Authority banned 13 dark patterns in its **Guidelines for the Prevention and Regulation of Dark Patterns, 2023**. Checking pages by hand does not scale."),
  p("**Goal:** build a tool that reads a product page and points out each manipulative line and its type, accurately enough to use on real Indian shopping sites, not only on a clean benchmark."),

  h1("2", "Data"),
  p("**Benchmark.** We used the e-commerce dark pattern dataset of Yada et al. (2022), built from the Mathur et al. crawl: 2,356 text snippets from real shopping sites, half dark patterns and half ordinary text (Table 1).", { keepNext: true }),
  typesTable,
  caption("Table 1.", "Dark pattern types in the benchmark."),
  p("**Indian shopping text.** The benchmark comes from Western sites, so we added (a) 368 Indian-style training lines generated from templates (rupee prices, delivery offers, ratings, product specifications), plus 32 separately written held-out lines; and (b) **171 lines mined from 17 real product pages** on Amazon.in, Snapdeal and six Indian brand stores: every line the first model scored above 0.3 or that used dark-pattern keywords, labelled by hand (11 dark, 160 not)."),
  p("**Real-page evaluation.** We labelled every line of 16 more product pages: 5 dev pages for tuning and 5 test pages from Amazon.in and Snapdeal (9 dark lines), plus 6 pages from **brand stores never used in training or tuning** (3 dark lines). None of these pages were used for mining."),

  h1("3", "Method"),
  p("**Baselines.** TF-IDF vectors (unigrams and bigrams) with Logistic Regression and a Linear SVM."),
  p("**Model.** DistilBERT (Sanh et al., 2019), a compressed BERT that keeps about 97% of its language understanding at 40% smaller size, reads each word in context: \"only\" is pressure in \"only 3 left\" but harmless in \"only available in cotton\". Our first model had 2 classes plus a separate TF-IDF type classifier. The final model has **one head with 6 classes**, not a dark pattern plus the five types, so it can say \"none of these\" and its two decisions cannot disagree. P(dark) is 1 minus P(not a dark pattern). Training: 3 epochs, AdamW, learning rate 2e-5, batch 16, 64 tokens."),
  p("**Honest splits.** A random split puts text from the same website in both training and test. The final benchmark split keeps each website on one side (GroupShuffleSplit on page id), so test scores measure unseen sites."),
  p("**From a page to lines.** The server downloads the page and keeps the visible text of each block element (paragraph, list item, table cell) as one line, joining inline tags so sentences are not split into fragments. It skips hidden text, dropdown lists, template placeholders, customer reviews (written by shoppers, not the seller) and lines with fewer than two words. On the Amazon.in phone page mentioned in the abstract, this cut 690 messy lines to 164 clean ones."),
  p("**Decision threshold.** Only 1 to 2% of the lines on a real page are dark patterns, against 50% in training, so a 0.5 cut-off raises far too many false alarms. A line is flagged when P(dark) > 0.97, chosen on the dev pages."),

  h1("4", "Results"),
    benchmarkTable,
  caption("Table 2.", "Dark pattern vs. not on the benchmark. The first three rows use a random split; the final model is tested on unseen websites."),
  p(`The final model reaches ${pct(o.f1)} F1 on websites it never saw and names the type correctly ${pct(T.accuracy)} of the time (macro F1 ${T.macro_f1.toFixed(3)}, up from ${NOMINE.category_on_dark_test.macro_f1.toFixed(3)} without the mined lines). It gets all 32 held-out Indian lines right. Trained and tested on a random split instead, the same setup scores ${pct(RANDOM.original_test.f1)}, so the random split did not inflate the earlier numbers.`),
  p("**Real product pages.** Table 3 follows the full pipeline, from page to verdict, through each change.", { keepNext: true }),
  realTable,
  caption("Table 3.", "Real-page results. \"Found\" counts labelled dark lines that were flagged; false alarms are flagged lines that are not dark patterns."),
  p("Page parsing removed most false alarms, the threshold removed most of the rest, and the 6-class model and mined data raised recall on the test pages back to every dark pattern. On the Amazon.in page that started this work, flags fell from 92 to 2 (\"Order within 8 hrs 40 mins\" and \"100K+ recent orders from this brand\"). **Caveats:** the evaluation sets are small (12 dark lines in total); the 0.97 threshold was kept, rather than re-tuned on dev pages that share sites with the mined data, after seeing the unseen-site results, so those results are optimistic; and the labels follow our written guide, which counts purchase counts such as \"recent orders\" as Social Proof."),

  h1("5", "System"),
    image("architecture.png", 640, 115),
  caption("Figure 1.", "System architecture and data flow for one scan."),
  image("dashboard_screenshot.png", 380, 273),
  caption("Figure 2.", "The dashboard: where each dark pattern sits on the page, the verdict, and a chart by type."),
  p("The dashboard states the main result in words, marks where each dark pattern sits on the page, explains why each line was flagged with its closest CCPA 2023 category, and lets people mark a mistake as \"Not a dark pattern\" (with Undo). It follows the system light or dark appearance, meets the WCAG AA contrast minimum of 4.5:1 for text, supports larger text, reduced motion and keyboard use, and was audited against Apple's Human Interface Guidelines. It is built with PyTorch and Hugging Face Transformers (training), FastAPI, requests and BeautifulSoup (server) and plain HTML, CSS and JavaScript (dashboard), and ships as a Docker image. When hosted publicly, the server refuses links to private network addresses.", { keepNext: true }),
  p("**To run it:**", { keepNext: true, spacing: { after: 40 } }),
  code("pip install -r app/requirements.txt"),
  code("python app/train.py     # trains the model (about 5 minutes on a laptop GPU)"),
  code("python app/server.py    # then open http://localhost:8000", true),

  h1("6", "Limitations and future work"),
  bullet("**Text only.** Visual tricks such as pre-ticked boxes, tiny decline buttons and hidden costs need a vision model trained on page screenshots."),
  bullet("**Wording, not truth.** The model flags pressure language; it cannot tell whether \"Only 2 left\" is false."),
  bullet("**Small real-page sets.** A few hundred hand-labelled Indian pages, including Hinglish and regional languages, would give firmer numbers."),
  bullet("**Page coverage.** Sites that block bots or render with JavaScript need pasted text; a browser extension would read the page as the shopper sees it."),

  h1("7", "Conclusion"),
  p(`A 6-class DistilBERT model detects dark patterns in shopping text with ${pct(o.f1)} F1 on unseen websites. Testing on real Indian product pages showed that benchmark accuracy was not enough: page parsing, a threshold matched to real pages and retraining on real mistakes took the system from 42.8 false alarms per page to none on the test pages. The dashboard makes the result usable by shoppers, and by regulators enforcing the CCPA 2023 guidelines.`),

  h1("", "References"),
  ...[
    "Mathur, A., Acar, G., Friedman, M. J., Lucherini, E., Mayer, J., Chetty, M., & Narayanan, A. (2019). Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites. Proc. ACM on Human-Computer Interaction (CSCW).",
    "Yada, Y., Feng, J., Matsumoto, T., Fukushima, N., Kido, F., & Yamana, H. (2022). Dark Patterns in E-Commerce: A Dataset and Its Baseline Evaluations. IEEE International Conference on Big Data.",
    "Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.",
    "Central Consumer Protection Authority, Government of India (2023). Guidelines for the Prevention and Regulation of Dark Patterns, 2023.",
  ].map((t, i) => new Paragraph({ spacing: { after: 30 }, indent: { left: 340, hanging: 340 }, children: [
    new TextRun({ text: `[${i + 1}] ${t}`, size: 17 })] })),
  new Paragraph({ spacing: { before: 60 }, children: [
    new TextRun({ text: "Source code: ", bold: true, size: 17 }),
    new ExternalHyperlink({ link: "https://github.com/ketanabhishek8/Dark-Pattern-Website-Detector",
      children: [new TextRun({ text: "https://github.com/ketanabhishek8/Dark-Pattern-Website-Detector", style: "Hyperlink", size: 17 })] }),
  ] }),
];

const doc = new Document({
  title: "Dark Pattern Detector", creator: "Abhishek Joshi, Shaurya Ghorpade",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 38, bold: true, color: NAVY, font: FONT }, paragraph: { spacing: { after: 40 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: NAVY, font: FONT }, paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 460, hanging: 230 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
