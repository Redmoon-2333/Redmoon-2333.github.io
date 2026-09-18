(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 年份 */
  document.querySelectorAll("[data-year]").forEach(function (el) {
    el.textContent = new Date().getFullYear();
  });

  /* Keep wide tables and figure notes readable without changing article Markdown. */
  var article = document.querySelector("article.post");
  if (article) {
    var prose = article.querySelector(".prose");
    article.closest("main").classList.add("reading-wrap");
    prose.querySelectorAll("table").forEach(function (table, index) {
      var wrapper = document.createElement("div");
      wrapper.className = "table-scroll";
      if (table.rows[0] && table.rows[0].cells.length >= 4) wrapper.classList.add("is-wide");
      if (table.rows[0] && table.rows[0].cells.length <= 2) wrapper.classList.add("is-compact");
      wrapper.tabIndex = 0;
      wrapper.setAttribute("role", "region");
      wrapper.setAttribute("aria-label", "表格 " + (index + 1));
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
    prose.querySelectorAll("blockquote").forEach(function (quote) {
      var previous = quote.previousElementSibling;
      if (previous && previous.querySelector("img") && /^图\s*\d/.test(quote.textContent.trim())) {
        quote.classList.add("figure-note");
      }
    });

    var headings = Array.from(prose.querySelectorAll("h2"));
    if (headings.length >= 3) {
      var sidebar = document.createElement("aside");
      sidebar.className = "post-toc";
      var details = document.createElement("details");
      var summary = document.createElement("summary");
      summary.textContent = "本文目录";
      details.appendChild(summary);
      var nav = document.createElement("nav");
      nav.setAttribute("aria-label", "文章章节");
      var list = document.createElement("ol");
      var links = headings.map(function (heading, index) {
        if (!heading.id) heading.id = "section-" + (index + 1);
        var item = document.createElement("li");
        var link = document.createElement("a");
        link.href = "#" + encodeURIComponent(heading.id);
        link.textContent = heading.textContent;
        link.addEventListener("click", function () {
          if (!desktop.matches) details.open = false;
        });
        item.appendChild(link);
        list.appendChild(item);
        return link;
      });
      nav.appendChild(list);
      details.appendChild(nav);
      sidebar.appendChild(details);
      article.appendChild(sidebar);
      article.classList.add("has-toc");
      var desktop = window.matchMedia("(min-width: 1051px)");
      details.open = desktop.matches;
      desktop.addEventListener("change", function (event) { details.open = event.matches; });

      var queued = false;
      function markSection() {
        queued = false;
        var current = 0;
        headings.forEach(function (heading, index) {
          if (heading.getBoundingClientRect().top <= 120) current = index;
        });
        links.forEach(function (link, index) {
          if (index === current) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      }
      window.addEventListener("scroll", function () {
        if (!queued) {
          queued = true;
          window.requestAnimationFrame(markSection);
        }
      }, { passive: true });
      markSection();
    }
  }

  /* 滚动显现 */
  var items = document.querySelectorAll(".reveal-on-scroll");
  if (!("IntersectionObserver" in window) || reduce) {
    items.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    items.forEach(function (el) { io.observe(el); });
  }

  /* 签名元素：checkpoint 打字日志 */
  var el = document.getElementById("log-text");
  if (!el) return;
  var LINES = [
    "nvidia-smi --query-gpu=memory.total → 8188 MiB · 够用，开训",
    "[phase 0] LangGraph 收尾中 · miniGPT 手写进行时",
    "git push origin main   # checkpoint 已保存",
    "next: CS336 assignment 1 · BPE tokenizer from scratch",
    "loss ↓ · curiosity ↑"
  ];
  if (reduce) { el.textContent = LINES[0]; return; }

  var li = 0, ci = 0, deleting = false;
  function tick() {
    if (document.hidden) { setTimeout(tick, 800); return; }
    var line = LINES[li];
    if (!deleting) {
      ci++;
      el.textContent = line.slice(0, ci);
      if (ci === line.length) { deleting = true; setTimeout(tick, 2800); return; }
      setTimeout(tick, 30);
    } else {
      ci -= 3;
      if (ci <= 0) {
        ci = 0; deleting = false; li = (li + 1) % LINES.length;
        el.textContent = "";
        setTimeout(tick, 420);
        return;
      }
      el.textContent = line.slice(0, ci);
      setTimeout(tick, 11);
    }
  }
  tick();
})();
