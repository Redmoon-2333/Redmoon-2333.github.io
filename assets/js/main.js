(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 年份 */
  document.querySelectorAll("[data-year]").forEach(function (el) {
    el.textContent = new Date().getFullYear();
  });

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
