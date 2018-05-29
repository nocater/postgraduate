# Git 常用命令
## git diff
git diff <filename> 比较*工作区*和*暂存区*的文件  
git diff --cached <filename> 比较*暂存区*和版本库的文件  
git diff --staged <filename> 比较*暂存区*和版本库的文件  
git diff HEAD 比较*工作区*和*版本库*的文件
patch 恢复操作

## git clone
git clone <git_url> master <local_floder> 克隆项目  
git colne <git_url> -b 远程分支 <local_floder> 拉取特定分支  

## git branch
git branch new_branch 创建新的分支  
git branch -l 显示所有分支  
git branch -v 显示所有分支及最后提交信息  
git branch -d branch 删除分支  
git branch -D 强制删除未合并的分支  

## git status    
git status -s 简要显示状态信息

## git add  
git add .  提交所有到*暂存区*  
git add -u 将被追踪的文件全部提交到*暂存区*  
git add -A 将改动和新加文件提交到*暂存区*  
git add -f 强制提交  
git add -i 交互提交  

## git checkout
git checkout 汇总显示*工作区*、*暂存区* 与HEAD的差异[显示与origin/master一致与否]  
git checkout [<brach\>] 切换分支  
**git checout <file> 目标是工作区的文件**  
git checkout [-q] [<commit>] [--] <paths>...  使用<commit\>版本覆盖工作区 如果省略<commit\>则指定*暂存区*进行覆盖  
git checkout – filename 撤销git .add <filename> 命令[使用*暂存区*文件覆盖*工作区*修改]  
git checkout branch – filename 将branch所指向的提交中的filename替换*暂存区*和*工作区*中相应的文件。
git checkout . 重置*工作区*，使用*暂存区*进行覆盖  
git checkout HEAD . 重置*工作区*，使用*版本库*进行覆盖  
git checkout -b <new_branch\> <remote\>/<branch\>  拉取远程分支并创建新分支

## git reset
<paths\> 有：重置指定路径文件 无：重置引用  
git reset [HEAD] 重置*暂存区*  
git reset HEAD --hard 使用*版本库*重置*暂存区*和*工作区*  
git reset HEAD --soft 仅重置引用，不重置*暂存区*和*工作区*
git reset [-mixed] HEAD^ 回退一次引用，并使用其覆盖*暂存区*

## git commit
git commit -a **不推荐使用** 将追踪文件直接提交  
git commit --amend 修补命令  
git commit -c 修改重用提交信息 -C直接使用不修改  
git commit -F 从文件读取

## git stash
git stash list 显示所有  
git stash save *message* 保存    
git stash pop pop Stash  
git stash apply 恢复不删除  
git stash drop 删除  
git stash clear 清空Stash

## git tag
git tag  显示所有标签  
git tag -a *tagtName* -m *tagMessage* 创建tag  
git show *tagNmae*  显示tag详细信息  
git tag -s GPG密钥签署  
git tag -v GPG验证  
git push origin *tagName* 推送标签到远程版本库    
git push <remote_url>  :<tagname\> 删除远程版本库tag
+ 里程碑共享，必须显式的推送。即在推送命令的参数中，标明要推送哪个里程碑。
+ 执行获取或拉回操作，自动从远程版本库获取新里程碑，并在本地版本库中创建。
+ 如果本地已有同名的里程碑，默认不会从上游同步里程碑，即使两者里程碑的指向是不同的。

## git push
git push <remote> <new_branch> 创建远程分支  
git remote -v 显示remote信息
git remote set-url *remoteName* 修改remote信息
git remote add 添加remote

## git pull
git pull --rebase  设置变基而不是合并
git config branch.<branchname>.rebase true 设置pull默认采用rebase  


## Tips
git rm --cache <filename\> 取消文件追踪  
git reflog show master | head -5 显示  
HEAD^ HEAD~3 master@{n} 引用表示

### Markdown 的使用
\# 一级标题  \#\#二级标题  
两个空格换行  
\.\+\-序列使用   
\*斜体\* \_斜体\_  
\*\*粗体\*\*，\_\_粗体\_\_
