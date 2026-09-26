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
  const klass = document.getElementById("list-class");
  const walls = document.getElementById("list-walls");
  const hiBtn = document.getElementById("list-highlighted");
  if (!q) return;
  const cards = Array.prototype.slice.call(document.querySelectorAll("[data-pair]"));
  const sections = Array.prototype.slice.call(document.querySelectorAll("[data-feedback-section]"));
  const empty = document.getElementById("browse-empty");
  let onlyHi = false;

  function chosen(root) {
    if (!root) return [];
    return Array.prototype.map.call(root.querySelectorAll("input:checked"), function (input) {
      return input.value;
    });
  }

  function matches(picked, value) {
    return !picked.length || picked.indexOf(value) !== -1;
  }

  function caption(root) {
    const btn = root.querySelector(".multi-btn");
    if (!btn) return;
    const picked = chosen(root);
    if (!picked.length) {
      btn.textContent = root.getAttribute("data-blank") || "";
      return;
    }
    const names = picked.map(function (value) {
      const input = root.querySelector('input[value="' + value + '"]');
      return input ? input.parentNode.textContent.trim() : value;
    });
    btn.textContent = names.join(", ");
  }

  function closeMenus(except) {
    document.querySelectorAll(".multi").forEach(function (menu) {
      if (menu === except) return;
      const panel = menu.querySelector(".multi-panel");
      const btn = menu.querySelector(".multi-btn");
      if (panel) panel.hidden = true;
      if (btn) btn.setAttribute("aria-expanded", "false");
    });
  }

  function render() {
    const text = (q.value || "").trim().toLowerCase();
    const lab = chosen(label);
    const dom = chosen(domination);
    const setl = chosen(settle);
    const cp = chosen(klass);
    const wall = chosen(walls);
    cards.forEach(function (card) {
      const hay = card.getAttribute("data-pair") || "";
      const matchText = !text || hay.indexOf(text) !== -1;
      const matchLab = matches(lab, card.getAttribute("data-label"));
      const matchDom = matches(dom, card.getAttribute("data-domination"));
      const matchSet = matches(setl, card.getAttribute("data-settle"));
      const matchCp = matches(cp, card.getAttribute("data-class-pair"));
      const matchWalls = matches(wall, card.getAttribute("data-walls"));
      const matchHi = !onlyHi || card.classList.contains("highlighted");
      card.classList.toggle("hidden", !(matchText && matchLab && matchDom && matchSet && matchCp && matchWalls && matchHi));
    });
    let any = false;
    sections.forEach(function (section) {
      const n = section.querySelectorAll("[data-pair]:not(.hidden)").length;
      const count = section.querySelector("[data-browse-count]");
      if (count) count.textContent = String(n);
      const top = document.querySelector('[data-top-count="' + section.getAttribute("data-feedback-section") + '"]');
      if (top) top.textContent = String(n);
      const show = n > 0;
      section.classList.toggle("hidden", !show);
      if (show) any = true;
    });
    if (empty) empty.classList.toggle("hidden", any || !sections.length);
  }

  q.addEventListener("input", render);
  document.addEventListener("click", function (event) {
    const btn = event.target.closest(".multi-btn");
    if (btn) {
      const menu = btn.closest(".multi");
      const panel = menu.querySelector(".multi-panel");
      const willOpen = panel.hidden;
      closeMenus(willOpen ? menu : null);
      panel.hidden = !willOpen;
      btn.setAttribute("aria-expanded", willOpen ? "true" : "false");
      return;
    }
    if (!event.target.closest(".multi")) closeMenus(null);
  });
  document.addEventListener("change", function (event) {
    const menu = event.target.closest(".multi");
    if (!menu || !q) return;
    caption(menu);
    render();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeMenus(null);
  });
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
      const saved = localStorage.getItem(KEY);
      if (saved === "color" || saved === "decisive") return saved;
      return "bw";
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

(function () {
  const HELP = {
    label: "Class of the black-and-white plot after generation 100, allowing the later row to slide. fixed: the line has stopped. cycle: it repeats in place. drift: it repeats while travelling. near_cycle: there is a real repeat that is not exact. aperiodic: the whole row does not come back.",
    period: "Lag of the first accepted repeat, in generations. Shown as none when the label is aperiodic.",
    shift: "How many cells the later row was rolled to make that repeat. 0 means it matches in place.",
    residual: "Leftover disagreement at that period after the best roll. An exact orbit is about 0.",
    residual_unshifted: "Disagreement at the same lag with no roll, so a sliding pattern still looks different.",
    min_shift_hamming: "Smallest shift-aware disagreement among the lags that were searched.",
    min_shift_lag: "Lag at which that smallest disagreement occurs.",
    mean_hamming: "Mean fraction of cells that flip from one row to the next, over the whole run.",
    std_hamming: "Standard deviation of that row-to-row activity.",
    mean_hamming_after_burn_in: "The same activity mean, using only rows after generation 100.",
    burn_in: "Generations left out of the Hamming label. The label starts after this row.",
    max_lag: "Longest lag searched, about one third of the post-burn-in length.",
    domination: "Who holds the color plot over the last 100 generations. dominate_complete: one rule has every cell. dominate: one rule has at least 90%, but not all. neutral: both rules are present, but fewer than 2% of cell-steps are decisive, so the arm choice barely matters. coexist: both rules are present and they compete.",
    domination_occupancy: "The occupancy tag before neutral is applied: dominate_complete, dominate, or coexist.",
    dominant_rule: "a or b when one rule wins the occupancy test. none when the tag is coexist or neutral.",
    rule_share_a: "Fraction of cell-steps in the last 100 generations whose TA used rule A.",
    rule_share_b: "Fraction of those cell-steps whose TA used rule B.",
    decisive_frac: "Fraction of those cell-steps where the two rules disagree on the neighbourhood, so the arm choice decided the next state.",
    decisive_share_a: "Among decisive cell-steps, the fraction that used rule A. none when no cell-step was decisive.",
    settle: "When the black-and-white pattern at the bottom is already in place. early: by generation 100. late: after generation 100. tail_only: settled_at is only the fallback at the start of the tail, so the plot never showed the pattern locking in. never: the bottom itself is aperiodic.",
    settle_raw: "The settle tag before tail_only replaces that fallback late. early, late, or never.",
    settled_at: "First generation, stepped by 50, from which the rest of the plot is at least as clean as the bottom. none when settle is never.",
    late_label: "Hamming label of the last 167 rows only, the bottom of the plot.",
    late_period: "Period of that bottom window. none when the bottom is aperiodic.",
    arm_flip_rate: "Fraction of cells that switch arm in a generation, averaged over the last 100 generations.",
    freeze_gen: "Generation of the last arm switch anywhere on the ring. 0 means the arms never switched. Equal to the run length means a TA was still switching on the last step.",
    ta_confidence: "Mean depth inside the chosen arm over the last 100 generations. 0 sits on the arm boundary, 1 is the deepest state.",
    arm_walls_initial: "Number of boundaries between rule A and rule B around the ring at generation 0.",
    arm_walls_final: "The same count on the last row. Near 100 on a 201-cell ring is salt-and-pepper. A handful means territories.",
    same_arm_as_initial: "Fraction of cells whose final arm is the same arm they started with.",
    grid_size: "Number of cells on the ring.",
    generations: "Steps after the initial row.",
    feedback: "majority or minority neighbourhood feedback for this run.",
    feedback_radius: "Radius, in cells, of the neighbourhood whose agreement or disagreement is the feedback.",
    seed: "Random seed for the initial row and the TA states."
  };

  const tip = document.createElement("div");
  tip.className = "metric-tip";
  tip.hidden = true;
  tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);

  function show(btn) {
    const text = HELP[btn.getAttribute("data-metric-tip")];
    if (!text) return;
    tip.textContent = text;
    tip.hidden = false;
    const rect = btn.getBoundingClientRect();
    const margin = 8;
    let left = rect.left;
    let top = rect.bottom + 6;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
    const box = tip.getBoundingClientRect();
    if (box.right > window.innerWidth - margin) {
      left = Math.max(margin, window.innerWidth - box.width - margin);
      tip.style.left = left + "px";
    }
    if (box.bottom > window.innerHeight - margin) {
      tip.style.top = Math.max(margin, rect.top - box.height - 6) + "px";
    }
  }

  function hide() {
    tip.hidden = true;
  }

  document.addEventListener("mouseover", function (event) {
    const btn = event.target.closest("[data-metric-tip]");
    if (btn) show(btn);
  });
  document.addEventListener("mouseout", function (event) {
    const btn = event.target.closest("[data-metric-tip]");
    if (!btn) return;
    const next = event.relatedTarget;
    if (next && btn.contains(next)) return;
    hide();
  });
  document.addEventListener("focusin", function (event) {
    const btn = event.target.closest("[data-metric-tip]");
    if (btn) show(btn);
  });
  document.addEventListener("focusout", hide);
  window.addEventListener("scroll", hide, true);
})();
