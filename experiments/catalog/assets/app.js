(function () {
  const search = document.getElementById("catalog-search");
  const feedback = document.getElementById("catalog-feedback");
  const klass = document.getElementById("catalog-class");
  const out = document.getElementById("catalog-results");
  if (!search || !out || typeof CATALOG === "undefined") return;

  function href(item) {
    return item.feedback + "/" + item.class_pair + "/rule" + item.rule_a + "_vs_rule" + item.rule_b + ".html";
  }

  function render() {
    const q = (search.value || "").trim().toLowerCase();
    const fb = feedback ? feedback.value : "";
    const cp = klass ? klass.value : "";
    const nums = q.match(/\d+/g);
    const rows = CATALOG.filter(function (item) {
      if (fb && item.feedback !== fb) return false;
      if (cp && item.class_pair !== cp) return false;
      if (!q) return true;
      const hay = (
        "rule " + item.rule_a + " vs rule " + item.rule_b + " " +
        item.class_pair + " " + item.feedback + " " +
        (item.label || "") + " " + (item.domination || "") + " " +
        (item.settle || "") + " " + (item.taxonomy || "")
      ).toLowerCase();
      if (hay.indexOf(q) !== -1) return true;
      if (!nums) return false;
      return nums.every(function (n) {
        return String(item.rule_a) === n || String(item.rule_b) === n;
      });
    }).slice(0, 200);

    if (!q && !fb && !cp) {
      out.innerHTML = "<p class='lede'>Type a rule number (or pair) to jump into the catalog.</p>";
      return;
    }
    if (!rows.length) {
      out.innerHTML = "<p class='missing'>No pairs match.</p>";
      return;
    }
    out.innerHTML = rows.map(function (item) {
      const extra = [
        item.label, item.domination, item.settle
      ].filter(Boolean).join(" · ");
      const suffix = extra ? " · " + extra : "";
      return (
        "<a href='" + href(item) + "'>" +
        item.feedback + " / " + item.class_pair +
        " · rule " + item.rule_a + " vs " + item.rule_b + suffix +
        "</a>"
      );
    }).join("");
  }

  search.addEventListener("input", render);
  if (feedback) feedback.addEventListener("change", render);
  if (klass) klass.addEventListener("change", render);
})();

(function () {
  const q = document.getElementById("list-filter");
  const label = document.getElementById("list-label");
  const domination = document.getElementById("list-domination");
  const settle = document.getElementById("list-settle");
  const hiBtn = document.getElementById("list-highlighted");
  if (!q) return;
  const cards = Array.prototype.slice.call(document.querySelectorAll("[data-pair]"));
  let onlyHi = false;

  function render() {
    const text = (q.value || "").trim().toLowerCase();
    const lab = label ? label.value : "";
    const dom = domination ? domination.value : "";
    const setl = settle ? settle.value : "";
    cards.forEach(function (card) {
      const hay = card.getAttribute("data-pair") || "";
      const matchText = !text || hay.indexOf(text) !== -1;
      const matchLab = !lab || card.getAttribute("data-label") === lab;
      const matchDom = !dom || card.getAttribute("data-domination") === dom;
      const matchSet = !setl || card.getAttribute("data-settle") === setl;
      const matchHi = !onlyHi || card.classList.contains("highlighted");
      card.classList.toggle("hidden", !(matchText && matchLab && matchDom && matchSet && matchHi));
    });
  }

  q.addEventListener("input", render);
  if (label) label.addEventListener("change", render);
  if (domination) domination.addEventListener("change", render);
  if (settle) settle.addEventListener("change", render);
  if (hiBtn) {
    hiBtn.addEventListener("click", function () {
      onlyHi = !onlyHi;
      hiBtn.classList.toggle("active", onlyHi);
      render();
    });
  }
})();

(function () {
  const KEY = "catalogGalleryView";

  function current() {
    try {
      return localStorage.getItem(KEY) === "color" ? "color" : "bw";
    } catch (e) {
      return "bw";
    }
  }

  function setThumb(img, view) {
    if (!img) return;
    const next = img.getAttribute("data-" + view) || img.getAttribute("data-bw");
    if (next) img.src = next;
    img.setAttribute("data-view", view);
  }

  function markButtons(root, attr, view) {
    root.querySelectorAll("[" + attr + "]").forEach(function (btn) {
      const on = btn.getAttribute(attr) === view;
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function setCardView(card, view) {
    setThumb(card.querySelector(".gallery-thumb"), view);
    markButtons(card, "data-item-view", view);
  }

  function applyAll(view) {
    document.querySelectorAll(".card").forEach(function (card) {
      if (card.querySelector(".gallery-thumb")) setCardView(card, view);
    });
    markButtons(document, "data-gallery-view", view);
  }

  function setAll(view) {
    try {
      localStorage.setItem(KEY, view);
    } catch (e) {}
    applyAll(view);
  }

  document.addEventListener("click", function (event) {
    const itemBtn = event.target.closest("[data-item-view]");
    if (itemBtn) {
      event.preventDefault();
      event.stopPropagation();
      const card = itemBtn.closest(".card");
      if (card) setCardView(card, itemBtn.getAttribute("data-item-view"));
      return;
    }
    const allBtn = event.target.closest("[data-gallery-view]");
    if (!allBtn) return;
    event.preventDefault();
    setAll(allBtn.getAttribute("data-gallery-view"));
  });

  applyAll(current());
})();
