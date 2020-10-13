---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: AutoPrompt
subtitle: Automatic Prompt Construction for Masked Language Models
callouts: paper
---

Welcome to the webpage for {{site.data.paper.description}}.

{{site.data.paper.abstract}}

## Paper

We published the paper at the [{{site.data.paper.venue.name}}]({{site.data.paper.venue.link}}).

<div class="block">
    <a class="button" href="{{site.url}}{{site.baseurl}}{{site.data.paper.pdf}}">Download PDF</a>
</div>

<div class="block">
<pre>
{{site.data.paper.citation-}}
</pre>
</div>

## Authors

<div class="columns  is-multiline">
{% for author in site.data.paper.authors %}
<div class="column is-2-desktop is-3-tablet is-6-phone">
    <div class="card">
        <header class="card-header">
            <p class="card-header-title is-centered">
                {% if author.website %}
                <a href="{{author.website}}">{{author.name}}</a>
                {% else %}
                {{author.name}}
                {% endif  %}
            </p>
        </header>
        <div class="card-image">
        <figure class="image is-1by1">
            <img src="{{author.img}}" alt="{{author.name}}">
        </figure>
        </div>
        <div class="card-content">
            <div class="content">
                <p class="has-text-centered">{{author.aff}}</p>
            </div>
        </div>
    </div>
</div>
{% endfor %}
</div>