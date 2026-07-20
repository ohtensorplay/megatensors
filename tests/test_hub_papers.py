from megatensors._hub.mega_api import _paper_info_from_mega_payload


def test_daily_paper_payload_maps_to_client_model():
    paper = _paper_info_from_mega_payload({
        "arxiv_id": "2607.05394",
        "title": "Native MEGA paper",
        "summary": "Official abstract.",
        "authors": ["Ada Researcher"],
        "published_at": "2026-07-16T08:00:00Z",
        "sources": ["arxiv", "huggingface"],
        "project_url": "https://example.test/project",
        "huggingface": {"upvotes": 9, "comments": 3, "github_stars": 12},
        "submission": {
            "submitted_at": "2026-07-17T08:00:00Z",
            "submitter": {
                "handle": "mega",
                "display_name": "MEGA Official",
                "avatar_url": "https://example.test/mega.png",
            },
        },
    })

    assert paper.id == "2607.05394"
    assert paper.title == "Native MEGA paper"
    assert paper.authors[0].name == "Ada Researcher"
    assert paper.upvotes == 9
    assert paper.comments == 3
    assert paper.submitted_by.username == "mega"
    assert paper.project_page == "https://example.test/project"


def test_paper_detail_maps_linked_mega_resources():
    paper = _paper_info_from_mega_payload({
        "paper": {
            "arxiv_id": "2607.05394",
            "title": "Native MEGA paper",
            "summary": "Official abstract.",
            "official_authors": ["Ada Researcher"],
            "published_at": "2026-07-16T08:00:00Z",
            "metadata_source": "arxiv",
        },
        "repos": [
            {"repo_id": "mega/paper-model", "repo_type": "model"},
            {"repo_id": "mega/paper-dataset", "repo_type": "dataset"},
            {"repo_id": "mega/paper-space", "repo_type": "space"},
        ],
    })

    assert [repo.id for repo in paper.linked_models] == ["mega/paper-model"]
    assert [repo.id for repo in paper.linked_datasets] == ["mega/paper-dataset"]
    assert [repo.id for repo in paper.linked_spaces] == ["mega/paper-space"]
