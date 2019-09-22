---
layout: page
title: Jupyter Notebooks
description: Jupyter Notebooks of Julian Abbott
permalink: /jupyter_notebook/
---

Jupyter Notebook documentation  

<ul>
  {% for post in site.categories.jupyter_notebook %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
