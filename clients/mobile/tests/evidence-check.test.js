import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEvidenceCheckQuestion } from '../src/services/evidence-check.js';
test('evidence recipe preserves arbitrary user materials, conditions and formatting exactly',()=>{
  const original='  我的判断：这项实验仅在低温有效？\n另一个作者反对。\r\n🚀';
  const question=buildEvidenceCheckQuestion(original);
  assert.ok(question.includes(original));assert.match(question,/相反证据/);assert.match(question,/不能支持/);assert.match(question,/没有检索到不等于不存在/);assert.match(question,/不补造来源/);
});
test('recipe is idempotent and never silently truncates oversized text',()=>{
  const question=buildEvidenceCheckQuestion('自己的观点'.repeat(5000));
  assert.ok(question.includes('自己的观点'.repeat(5000)));assert.equal(buildEvidenceCheckQuestion(question),question);
});
test('empty input creates no canned question',()=>{for(const value of ['', '  ',null,{},undefined])assert.equal(buildEvidenceCheckQuestion(value),'');});
