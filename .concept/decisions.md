# Active Decisions (Cache)

**Status**: Applied | **Last Updated**: 2026-01-27

## Model & Architecture

### [DEC-001] Primary Model Selection
**Decision**: Use XLM-RoBERTa (Twitter sentiment fine-tuned)
**Rationale**: Best cross-lingual performance with efficient inference
**Impact**: Core sentiment analysis functionality
**Milestone**: MS-001
**Status**: Applied ✅

### [DEC-002] API Framework Choice
**Decision**: FastAPI with Uvicorn
**Rationale**: Modern async framework with automatic OpenAPI docs
**Impact**: REST API performance and developer experience
**Milestone**: MS-001
**Status**: Applied ✅

## Configuration

### [DEC-003] Default Batch Size
**Decision**: 32 texts per batch
**Rationale**: Balance between memory usage and throughput
**Impact**: Batch processing performance
**Milestone**: MS-002
**Status**: Applied ⚠️ (Hardware-dependent optimization pending)

## Quality Standards

### [DEC-004] Test Coverage Target
**Decision**: Minimum 80% code coverage
**Rationale**: Ensures code quality and reliability
**Impact**: Development workflow and CI/CD
**Milestone**: MS-001
**Status**: Applied ✅

---

**Legend**: ✅ Applied | ⚠️ Conditional | 🔄 In Progress