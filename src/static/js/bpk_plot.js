(function () {
  const pending = new Map();

  async function render(div) {
    const b64 = pending.get(div.id);
    if (b64 === undefined) return;
    pending.delete(div.id);
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    const fig = JSON.parse(await new Response(stream).text());
    await Plotly.newPlot(div, fig.data, fig.layout,
                         Object.assign({responsive: true}, fig.config || {}));
  }

  const observer = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (e.isIntersecting) { observer.unobserve(e.target); render(e.target); }
    }
  }, {rootMargin: "200px"});

  window.bpkPlot = function (id, b64) {
    pending.set(id, b64);
    const div = document.getElementById(id);
    if (div) observer.observe(div);
  };
  window.bpkRender = function (id, btn) {
    if (btn) { btn.disabled = true; btn.textContent = "Rendering …"; }
    setTimeout(() => render(document.getElementById(id)).then(() => btn && btn.remove()), 50);
  };
})();